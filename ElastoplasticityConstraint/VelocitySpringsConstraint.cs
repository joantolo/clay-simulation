using Particleworks.Computing;
using Particleworks.Math;
using Particleworks.Primitives;
using Particleworks.Simulation;
using UnityEngine;

namespace Particleworks.Fluids
{
    //////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// 
    /// Constraint of dynamic springs that (de)activate
    /// to model the elastoplastic behaviour.
    /// 
    /// </summary>
    //////////////////////////////////////////////////////////////////////////////////

    public class VelocitySpringsConstraint : ParticleConstraint
    {

        //== Constants =============================================================

        private const int DefaultSpringsGenerationFrequency = 1;

        //== Properties ============================================================

        public float VelocityThreshold
        {
            get { return m_VelocityThreshold; }
            set { m_VelocityThreshold = value; }
        }

        public int SpringsGenerationFrequency
        {
            get { return m_SpringsGenerationFrequency; }
            set { m_SpringsGenerationFrequency = value; }
        }

        public override bool IsPositionConstraint
        {
            get { return true; }
        }

        //== Members ===============================================================

        // ---- Params ----

        [Header("Velocity Springs Constraint")]

        [SerializeField]
        private float m_VelocityThreshold;

        [SerializeField]
        private int m_SpringsGenerationFrequency = DefaultSpringsGenerationFrequency;

        [SerializeField]
        private float m_DeltaLimit = 0.0f;

        // ---- Buffers ----

        private int m_NumConstraints;
        private int m_MaxConstraints;

        private ComputeDoubleDataBuffer<UIntVector2> m_ConstraintTuple;
        private ComputeDoubleDataBuffer<float> m_ConstraintDistance;
        private ComputeDataBuffer<int> m_ConstraintOrder;
        private ComputeDataBuffer<int> m_ConstraintStencil;
        private ComputeDataBuffer<UIntVector2> m_ConstraintLocalOrder;

        private PrefixScan.WorkData m_ConstraintPrefixScan;
        private Reduction.WorkData<int> m_ConstraintReduction;

        private PrefixScan.WorkData m_PerPartPrefixScan;
        private Reduction.WorkData<int> m_PerPartReductionData;

        private ComputeDataBuffer<int> m_PerPartCount;
        private ComputeDataBuffer<int> m_PerPartOffset;
        private ComputeDataBuffer<int> m_PerPartConstraints;

        private ComputeDoubleDataBuffer<float> m_PerPartLastUs;
        private ComputeDataBuffer<float> m_PerPartAverageLastUs;

        private ComputeDataBuffer<int> m_AppendCounter;

        private ComputeDataBuffer<float> m_DebugInfo;

        //== Methods ===============================================================

        // ---- Behaviour events ----

        protected override bool OnComponentAttached()
        {
            if (base.OnComponentAttached())
            {
                Program = new VelocitySpringsConstraintProgram();
                ResizeMaxConstraints(Simulation.MaxParticles * 100);
                m_PerPartCount = new ComputeDataBuffer<int>(Simulation.MaxParticles);
                m_PerPartOffset = new ComputeDataBuffer<int>(Simulation.MaxParticles);
                m_PerPartPrefixScan = new PrefixScan.WorkData(Simulation.MaxParticles);
                m_PerPartReductionData = new Reduction.WorkData<int>(Simulation.MaxParticles);
                m_PerPartLastUs = new ComputeDoubleDataBuffer<float>(3 * Simulation.MaxParticles * 4);
                m_PerPartAverageLastUs = new ComputeDataBuffer<float>(3 * Simulation.MaxParticles);
                m_AppendCounter = new ComputeDataBuffer<int>(1);
                m_DebugInfo = new ComputeDataBuffer<float>(Simulation.MaxParticles * 100 * 3);
                Simulation.OnBeginSubUpdate.AddListener(Priority, OnBeginSubUpdate);
                Simulation.OnParticlesRearranged.AddListener(Priority, OnParticlesRearranged);
                return true;
            }

            return false;
        }

        protected override void OnComponentDettached()
        {
            base.OnComponentDettached();
            Program.Dispose();
            m_ConstraintTuple.Dispose();
            m_ConstraintDistance.Dispose();
            m_ConstraintOrder.Dispose();
            m_ConstraintStencil.Dispose();
            m_ConstraintLocalOrder.Dispose();
            m_ConstraintPrefixScan.Dispose();
            m_ConstraintReduction.Dispose();
            m_PerPartConstraints.Dispose();
            m_PerPartCount.Dispose();
            m_PerPartOffset.Dispose();
            m_PerPartPrefixScan.Dispose();
            m_PerPartReductionData.Dispose();
            m_PerPartLastUs.Dispose();
            m_PerPartAverageLastUs.Dispose();
            m_AppendCounter.Dispose();
            m_DebugInfo.Dispose();
            Simulation.OnBeginSubUpdate.RemoveListener(Priority, OnBeginSubUpdate);
            Simulation.OnParticlesRearranged.RemoveListener(Priority, OnParticlesRearranged);
        }

        // ---- Constraint Events ----

        protected override void BindConstraintConstants(ComputeProgram program, bool velocities)
        {
            base.BindConstraintConstants(program, velocities);
            program.SetFloat("VelocityThreshold", m_VelocityThreshold);
            program.SetInt("NumSpringConstraints", m_NumConstraints);
            program.SetInt("MaxSpringConstraints", m_MaxConstraints);
            program.SetFloat("DeltaLimit", m_DeltaLimit);
        }

        protected void OnBeginSubUpdate()
        {
            UpdateVelocities();

            if (Simulation.CurrentFrame % m_SpringsGenerationFrequency != 0) return;

            // Build neighbourhoods.

            Simulation.SearchNeighbours();

            // Generate and update constraints.

            Simulation.BindGlobalConstants(Program);
            Simulation.BindPerGroupConstants(Program);
            BindConstraintConstants(Program, false);

            GenerateConstraints();
            UpdateConstraints();
            CollapseConstraints();
        }

        protected override void PrepareForPositionSolver()
        {
            // Generate and update constraints.

            Simulation.BindGlobalConstants(Program);
            Simulation.BindPerGroupConstants(Program);
            BindConstraintConstants(Program, false);
        }

        protected override bool SolvePositionConstraint()
        {
            if (m_NumConstraints == 0) return false;

            UpdateConstraints();

            Simulation.BindGlobalConstants(Program);
            Simulation.BindPerGroupConstants(Program);
            BindConstraintConstants(Program, false);

            Program.SetKernel("KerComputeDelta");
            Program.SetFrontBuffer("PartInfoR", Simulation.Data.Info);
            Program.SetFrontBuffer("PartPositionR", Simulation.Data.Position);
            Program.SetBuffer("PerPartOffsetR", m_PerPartOffset);
            Program.SetBuffer("PerPartCountR", m_PerPartCount);
            Program.SetBuffer("PerPartConstraintsR", m_PerPartConstraints);
            Program.SetFrontBuffer("ConstraintTupleR", m_ConstraintTuple);
            Program.SetBuffer("ConstraintStencilR", m_ConstraintStencil);
            Program.SetFrontBuffer("ConstraintDistanceR", m_ConstraintDistance);
            Program.SetFrontBuffer("PartDeltaRW", Simulation.Data.Delta);
            Program.LaunchAuto(Simulation.NumParticles);

            return true;
        }

        private void GenerateConstraints()
        {
            // Set append counter to current number of constraints.

            var counter = new[] { m_NumConstraints };
            m_AppendCounter.SetData(counter);

            // Generate new constraints.

            Program.SetKernel("KerGenerateConstraints");
            Program.SetFrontBuffer("PartInfoR", Simulation.Data.Info);
            Program.SetFrontBuffer("PartPositionR", Simulation.Data.Position);
            Program.SetFrontBuffer("PartVelocityR", Simulation.Data.Velocity);
            Program.SetBuffer("PartNeighboursR", Simulation.Data.Neighbours);
            Program.SetBuffer("PartNeighboursCountR", Simulation.Data.NeighboursCount);
            Program.SetBuffer("PerPartOffsetR", m_PerPartOffset);
            Program.SetBuffer("PerPartCountR", m_PerPartCount);
            Program.SetBuffer("PerPartConstraintsR", m_PerPartConstraints);
            Program.SetFrontBuffer("ConstraintTupleRW", m_ConstraintTuple);
            Program.SetFrontBuffer("ConstraintDistanceRW", m_ConstraintDistance);
            Program.SetFrontBuffer("PerPartLastUsRW", m_PerPartLastUs);
            Program.SetBuffer("PerPartAverageLastUsRW", m_PerPartAverageLastUs);
            Program.SetBuffer("AppendCounterRW", m_AppendCounter);
            Program.LaunchAuto(Simulation.NumParticles);

            // Update constraint count.

            m_AppendCounter.GetData(counter);
            var newNumConstraints = Mathf.Min(counter[0], m_MaxConstraints);
            //Debug.LogFormat("Generation : {0} => {1}", m_NumConstraints, newNumConstraints);
            m_NumConstraints = newNumConstraints;

            // If we have reached the maximum number of constraints, resize
            // buffers!

            //if (m_NumConstraints == m_MaxConstraints)
            //    ResizeMaxConstraints(m_MaxConstraints * 2);

            // Update constants.

            Program.SetInt("NumSpringConstraints", m_NumConstraints);
            Program.SetInt("MaxSpringConstraints", m_MaxConstraints);
        }

        private void UpdateVelocities()
        {
            Simulation.BindGlobalConstants(Program);
            Simulation.BindPerGroupConstants(Program);
            Program.SetKernel("KerUpdatePartLastVelocities");
            Program.SetFrontBuffer("PartInfoR", Simulation.Data.Info);
            Program.SetFrontBuffer("PartVelocityR", Simulation.Data.Velocity);
            Program.SetFrontBuffer("PerPartLastUsRW", m_PerPartLastUs);
            Program.SetBuffer("PerPartAverageLastUsRW", m_PerPartAverageLastUs);
            Program.LaunchAuto(Simulation.NumParticles);
        }

        private void UpdateConstraints()
        {
            if (m_NumConstraints == 0) return;

            // Update constraints and update stencil to determine which are still alive.

            Program.SetKernel("KerUpdateConstraints");
            Program.SetFrontBuffer("ConstraintTupleRW", m_ConstraintTuple);
            Program.SetFrontBuffer("ConstraintDistanceRW", m_ConstraintDistance);
            Program.SetFrontBuffer("PartPositionR", Simulation.Data.Position);
            Program.SetBuffer("PerPartAverageLastUsR", m_PerPartAverageLastUs);
            Program.SetBuffer("ConstraintStencilRW", m_ConstraintStencil);
            Program.LaunchAuto(m_NumConstraints);
        }

        private void CollapseConstraints()
        {
            if (m_NumConstraints == 0) return;
            var oldNumConstraints = m_NumConstraints;

            // Determine number of alive constraints and their order.

            m_NumConstraints = Reduction.ExecuteSum(m_ConstraintReduction, m_ConstraintStencil, oldNumConstraints);
            //Debug.LogFormat("Collapse : {0} => {1}", oldNumConstraints, numConstraints);
            PrefixScan.Execute(m_ConstraintPrefixScan, m_ConstraintStencil, m_ConstraintOrder, oldNumConstraints);

            // For all the previous constraints, rearrange them to their final location.

            Program.SetKernel("KerRearrangeConstraints");
            Program.SetFrontBuffer("ConstraintTupleR", m_ConstraintTuple);
            Program.SetFrontBuffer("ConstraintDistanceR", m_ConstraintDistance);
            Program.SetBackBuffer("ConstraintTupleRW", m_ConstraintTuple);
            Program.SetBackBuffer("ConstraintDistanceRW", m_ConstraintDistance);
            Program.SetBuffer("ConstraintStencilR", m_ConstraintStencil);
            Program.SetBuffer("ConstraintOrderR", m_ConstraintOrder);
            Program.LaunchAuto(oldNumConstraints);

            m_ConstraintTuple.SwapBuffers();
            m_ConstraintDistance.SwapBuffers();

            // Update constants.
            int changedConstraints = m_NumConstraints - oldNumConstraints;


            Program.SetInt("NumViscConstraints", m_NumConstraints);

            // Update particle constraints list

            UpdateParticleConstraints();
        }

        private void UpdateParticleConstraints()
        {
            // Initialize counts to zero.

            Fill.Execute(m_PerPartCount, 0, 0, Simulation.NumParticles);

            if (m_NumConstraints == 0) return;

            // Count constraints per particle and obtain their local offset within
            // the particle's list.

            Program.SetKernel("KerCountParticleConstraints");
            Program.SetFrontBuffer("ConstraintTupleR", m_ConstraintTuple);
            Program.SetBuffer("ConstraintLocalOrderRW", m_ConstraintLocalOrder);
            Program.SetBuffer("PerPartCountRW", m_PerPartCount);
            Program.LaunchAuto(m_NumConstraints);

            // Perform prefix scan to determine offsets to their final locations.

            PrefixScan.Execute(m_PerPartPrefixScan, m_PerPartCount, m_PerPartOffset, Simulation.NumParticles);

            // Finally, store the constraint indices wherever they were meant to.

            Program.SetKernel("KerStoreParticleConstraints");
            Program.SetFrontBuffer("ConstraintTupleR", m_ConstraintTuple);
            Program.SetBuffer("ConstraintLocalOrderR", m_ConstraintLocalOrder);
            Program.SetBuffer("PerPartOffsetR", m_PerPartOffset);
            Program.SetBuffer("PerPartConstraintsRW", m_PerPartConstraints);
            Program.LaunchAuto(m_NumConstraints);
        }

        private void OnParticlesRearranged()
        {
            if (m_NumConstraints == 0) return;

            Simulation.BindGlobalConstants(Program);
            Simulation.BindPerGroupConstants(Program);
            BindConstraintConstants(Program, false);

            // Update constraint indices and build removal stencil.

            Program.SetKernel("KerRearrangeParticles");
            Program.SetFrontBuffer("ConstraintTupleRW", m_ConstraintTuple);
            Program.SetBuffer("ConstraintStencilRW", m_ConstraintStencil);
            Simulation.BindParticleOrderBuffers(Program);
            Program.LaunchAuto(m_NumConstraints);

            // Update last velocities on rearranged particles.

            Program.SetKernel("KerRearrangeParticlesU");
            Program.SetFrontBuffer("PerPartLastUsR", m_PerPartLastUs);
            Program.SetBackBuffer("PerPartLastUsRW", m_PerPartLastUs);
            Simulation.BindParticleOrderBuffers(Program);
            Program.LaunchAuto(Simulation.NumParticles);
            m_PerPartLastUs.SwapBuffers();

            // Collapse constraints.

            //Debug.Log("OnParticlesRearranged");
            CollapseConstraints();
        }

        private void ResizeMaxConstraints(int newMaxConstraints)
        {
            if (m_MaxConstraints == newMaxConstraints)
                return;

            var oldConstraintTuple = m_ConstraintTuple;
            var oldConstraintDistance = m_ConstraintDistance;
            var oldConstraintOrder = m_ConstraintOrder;
            var oldConstraintStencil = m_ConstraintStencil;
            var oldConstraintLocalOrder = m_ConstraintLocalOrder;
            var oldPerPartConstraints = m_PerPartConstraints;

            var oldConstraintReduction = m_ConstraintReduction;
            var oldConstraintPrefixScan = m_ConstraintPrefixScan;

            Debug.LogFormat("Resized constraints buffer to {0}", newMaxConstraints);

            m_MaxConstraints = newMaxConstraints;
            m_ConstraintTuple = new ComputeDoubleDataBuffer<UIntVector2>(m_MaxConstraints);
            m_ConstraintDistance = new ComputeDoubleDataBuffer<float>(m_MaxConstraints);
            m_ConstraintOrder = new ComputeDataBuffer<int>(m_MaxConstraints);
            m_ConstraintStencil = new ComputeDataBuffer<int>(m_MaxConstraints);
            m_ConstraintLocalOrder = new ComputeDataBuffer<UIntVector2>(m_MaxConstraints);
            m_PerPartConstraints = new ComputeDataBuffer<int>(m_MaxConstraints);
            m_ConstraintPrefixScan = new PrefixScan.WorkData(m_MaxConstraints);
            m_ConstraintReduction = new Reduction.WorkData<int>(m_MaxConstraints);

            if (m_NumConstraints > 0)
            {
                Copy.Execute(oldConstraintTuple.FrontBuffer, 0, m_ConstraintTuple.FrontBuffer, 0, m_NumConstraints);
                Copy.Execute(oldConstraintDistance.FrontBuffer, 0, m_ConstraintDistance.FrontBuffer, 0, m_NumConstraints);
                Copy.Execute(oldPerPartConstraints, 0, m_PerPartConstraints, 0, m_NumConstraints);
            }

            Compute.SafelyDisposeBuffer(oldConstraintTuple);
            Compute.SafelyDisposeBuffer(oldConstraintDistance);
            Compute.SafelyDisposeBuffer(oldConstraintOrder);
            Compute.SafelyDisposeBuffer(oldConstraintStencil);
            Compute.SafelyDisposeBuffer(oldConstraintLocalOrder);
            Compute.SafelyDisposeBuffer(oldPerPartConstraints);

            if (oldConstraintReduction != null) oldConstraintReduction.Dispose();
            if (oldConstraintPrefixScan != null) oldConstraintPrefixScan.Dispose();
        }
    }
}