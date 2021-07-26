using Particleworks.Computing;
using Particleworks.Fluids;
using Particleworks.Math;
using Particleworks.Primitives;
using Particleworks.Simulation;
using Particleworks.Utils;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Assertions;

namespace Particleworks.Collisions
{
    ////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    ///
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////

    public class PointCloudCollisionConstraint : ParticleConstraint
    {
        //== Constants =============================================================

        public const int MaxContacts = 1;

        //== Properties ============================================================

        public int PrestabIters { get => m_PrestabIters; set => m_PrestabIters = value; }

        public IEnumerable<SimulationPointCloudCollider> Colliders { get => m_Colliders; }

        public float ContactRadius { get => m_ContactRadius; set => m_ContactRadius = value; }

        public float PartAvgeSpacing { get => m_PartAvgeSpacing; }

        public override bool IsPositionConstraint => true;

        public override bool IsVelocityConstraint => true;

        public override int Priority => Priorities.Behaviour + 3;

        public int NumGhostParts { get => m_NumGhostParts; }

        public int MaxGhostParts { get => m_MaxGhostParts; }

        public ComputeDataVector<float> GhostPartPosition { get => m_GhostPartPosition.FrontVector; }

        public ComputeDataVector<float> GhostPartPositionDelta { get => m_GhostPartPositionDelta.FrontVector; }

        public ComputeDataVector<int> GhostPartColliderId { get => m_GhostPartColliderId.FrontVector; }

        public ComputeDataVector<int> GhostPartType { get => m_GhostPartType.FrontVector; }

        public ComputeDataVector<float> GhostPartClosestDist { get => m_GhostPartClosestDist.FrontVector; }

        public ComputeDataVector<float> GhostPartClosestDir { get => m_GhostPartClosestDir.FrontVector; }

        public ComputeDataVector<int> GhostPartTetId { get => m_GhostPartTetId.FrontVector; }

        public ComputeDataVector<int> GhostPartClosestTriId { get => m_GhostPartClosestTriId.FrontVector; }

        public ComputeDataVector<float> GhostPartTriNTetBarycentricCoords { get => m_GhostPartTriNTetBarycentricCoords.FrontVector; }

        public ComputeDataVector<int> GhostPartIsSurface { get => m_GhostPartIsSurface.FrontVector; }

        public ComputeDataVector<int> ColliderGhostPartStart { get => m_ColliderGhostPartStart; }

        public ComputeDataVector<int> ColliderGhostPartCount { get => m_ColliderGhostPartCount; }

        public ComputeDataVector<float> ColliderFrictionCoeff { get => m_ColliderFrictionCoeff; }

        public ComputeDataVector<int> TetIndices { get => m_TetIndices; }

        public ComputeDataVector<int> SurfIndices { get => m_SurfIndices; }

        public ComputeDataVector<Vector3> Vertices { get => m_Vertices; }

        public ComputeDataVector<float> Weights { get => m_Weights; }

        //== Members ===============================================================

        // ---- Collision Settings ----

        [Header("Collisions Settings")]

        [SerializeField]
        private int m_PrestabIters = 3;

        [SerializeField]
        [RuntimeReadOnly]
        private int m_MaxGhostParts = 100000;

        [SerializeField]
        private float m_ContactRadius = 0.1f;

        [SerializeField]
        [RuntimeReadOnly]
        private float m_PartAvgeSpacing = 0.01f;

        [Header("Neighbourhood Parameters")]

        [SerializeField]
        private int m_MaxNeighbours = 60;

        [SerializeField]
        [RuntimeReadOnly]
        private int m_MaxNeighbourBuckets = 20000;

        // ---- Internal Data ----

        private int m_NumGhostParts;
        private bool m_IsReady;

        public ComputeDoubleDataBuffer<int> m_PartContactCount;                           // No. of contacts per-particle.
        private ComputeDoubleDataBuffer<int> m_PartContactInfo;                           // Contact info (validity, id, etc).
        private ComputeDoubleDataBuffer<float> m_PartContactPointsNPlanes;                // Contact location and planes.
        private ComputeDoubleDataBuffer<float> m_PartContactFrictionCoeff;                // Friction coefficients per-contact.

        public ComputeDoubleDataVector<float> m_GhostPartPosition;                        // Ghost particle position.
        public ComputeDoubleDataVector<float> m_GhostPartPositionDelta;                   // Ghost particle position delta (for updating the anchor).
        private ComputeDoubleDataVector<int> m_GhostPartColliderId;                       // Ghost particle collider ID.
        private ComputeDoubleDataVector<int> m_GhostPartType;                             // Type of particle. 0 - Outside, 1 - Boundary, 2 - Inside.
        private ComputeDoubleDataVector<float> m_GhostPartClosestDist;                    // Sparse sampling of the object's distance field.
        private ComputeDoubleDataVector<float> m_GhostPartClosestDir;                     // Sparse sampling of the object's distance field gradient.>
        private ComputeDoubleDataVector<int> m_GhostPartTetId;                            // Associated tetrahedron ID.
        private ComputeDoubleDataVector<int> m_GhostPartClosestTriId;                     // Associated closest surface triangle ID.
        private ComputeDoubleDataVector<float> m_GhostPartTriNTetBarycentricCoords;       // Associated triangle and tetrahedron barycentric coords.
        private ComputeDoubleDataVector<int> m_GhostPartIsSurface;                        // Type of particle 0 - Not in Surface, 1 - Surface.

        private ComputeDataVector<int> m_ColliderGhostPartStart;                          // Per-collider, first ghost particle.
        private ComputeDataVector<int> m_ColliderGhostPartCount;                          // Per-collider, number of ghost particles.
        private ComputeDataVector<float> m_ColliderFrictionCoeff;                         // Per-collider, friction coefficient.

        private ComputeDataVector<int> m_TetIndices;                                      // Underlying mesh tetrahedron indices.
        private ComputeDataVector<int> m_SurfIndices;                                     // Underlying mesh surface triangle indices.
        private ComputeDataVector<Vector3> m_Vertices;                                    // Underlying mesh vertices.
        private ComputeDataVector<Vector3> m_VerticesPrev;                                // Underlying mesh vertices (previous frame).
        private ComputeDataVector<Vector3> m_VerticesNext;                                // Underlying mesh vertices (next frame).
        private ComputeDataVector<float> m_Weights;                                       // Underlying mesh weights.

        private CountSort.WorkData m_NeighbourSortData;
        private ComputeDoubleDataBuffer<KeyValueInt> m_NeighbourSearchTuples;             // ID of tuple to order ghost particles.
        private ComputeDataBuffer<int> m_GhostPartNeighbourCellIndex;                     // ID of cell of neighbours to order ghost particles.
        private ComputeDataBuffer<int> m_PartGhostNeighboursCount;                        // Number of neighbours a ghost particle has.
        private ComputeDataBuffer<int> m_PartGhostNeighbours;                             // IDs of ghost particle neighbours.
        private ComputeDataBuffer<int> m_OrderNewToOld;                                   // Get the new ID of ghost particle.
        private ComputeDataBuffer<int> m_OrderOldToNew;                                   // Get the old ID of ghost particle.

        private readonly HashSet<SimulationPointCloudCollider> m_Colliders = new HashSet<SimulationPointCloudCollider>();

        //== Methods ===============================================================

        // ---- Unity events ----

        protected override void Update()
        {
            if (m_IsReady)
            {
                CreateGhostParticles();
                m_IsReady = false;
            }
        }

        // ---- Component Events ----

        protected override bool OnComponentAttached()
        {
            if (base.OnComponentAttached())
            {
                Program = new PointCloudCollisionConstraintProgram();

                m_PartContactCount = new ComputeDoubleDataBuffer<int>(Simulation.MaxParticles * MaxContacts);
                m_PartContactInfo = new ComputeDoubleDataBuffer<int>(Simulation.MaxParticles * MaxContacts);
                m_PartContactPointsNPlanes = new ComputeDoubleDataBuffer<float>(7 * Simulation.MaxParticles * MaxContacts);
                m_PartContactFrictionCoeff = new ComputeDoubleDataBuffer<float>(Simulation.MaxParticles * MaxContacts);

                m_GhostPartPosition = new ComputeDoubleDataVector<float>();
                m_GhostPartPositionDelta = new ComputeDoubleDataVector<float>();
                m_GhostPartColliderId = new ComputeDoubleDataVector<int>();
                m_GhostPartType = new ComputeDoubleDataVector<int>();
                m_GhostPartClosestDist = new ComputeDoubleDataVector<float>();
                m_GhostPartClosestDir = new ComputeDoubleDataVector<float>();
                m_GhostPartTetId = new ComputeDoubleDataVector<int>();
                m_GhostPartClosestTriId = new ComputeDoubleDataVector<int>();
                m_GhostPartTriNTetBarycentricCoords = new ComputeDoubleDataVector<float>();
                m_GhostPartIsSurface = new ComputeDoubleDataVector<int>();

                m_ColliderGhostPartStart = new ComputeDataVector<int>();
                m_ColliderGhostPartCount = new ComputeDataVector<int>();
                m_ColliderFrictionCoeff = new ComputeDataVector<float>();

                m_TetIndices = new ComputeDataVector<int>();
                m_SurfIndices = new ComputeDataVector<int>();
                m_Vertices = new ComputeDataVector<Vector3>();
                m_VerticesPrev = new ComputeDataVector<Vector3>();
                m_VerticesNext = new ComputeDataVector<Vector3>();
                m_Weights = new ComputeDataVector<float>();

                m_NeighbourSearchTuples = new ComputeDoubleDataBuffer<KeyValueInt>(m_MaxGhostParts);
                m_NeighbourSortData = new CountSort.WorkData(m_MaxGhostParts, m_MaxNeighbourBuckets);
                m_GhostPartNeighbourCellIndex = new ComputeDataBuffer<int>(m_MaxGhostParts);
                m_PartGhostNeighboursCount = new ComputeDataBuffer<int>(Simulation.MaxParticles);
                m_PartGhostNeighbours = new ComputeDataBuffer<int>(Simulation.MaxParticles * m_MaxNeighbours);
                m_OrderNewToOld = new ComputeDataBuffer<int>(m_MaxGhostParts);
                m_OrderOldToNew = new ComputeDataBuffer<int>(m_MaxGhostParts);

                Simulation.OnBeginUpdate.AddListener(Priority, UpdateColliders);
                Simulation.OnPositionsEstimated.AddListener(Priority, DetectContactsAndPrestabilize);
                Simulation.OnParticlesRearranged.AddListener(Priority, RearrangeContactsAndGhostParts);

                m_IsReady = true;

                return true;
            }

            return false;
        }

        protected override void OnComponentDettached()
        {
            base.OnComponentDettached();
            m_PartContactCount.Dispose();
            m_PartContactInfo.Dispose();
            m_PartContactPointsNPlanes.Dispose();
            m_PartContactFrictionCoeff.Dispose();

            m_GhostPartPosition.Dispose();
            m_GhostPartPositionDelta.Dispose();
            m_GhostPartColliderId.Dispose();
            m_GhostPartType.Dispose();
            m_GhostPartClosestDist.Dispose();
            m_GhostPartClosestDir.Dispose();
            m_GhostPartTetId.Dispose();
            m_GhostPartClosestTriId.Dispose();
            m_GhostPartTriNTetBarycentricCoords.Dispose();
            m_GhostPartIsSurface.Dispose();

            m_ColliderGhostPartStart.Dispose();
            m_ColliderGhostPartCount.Dispose();
            m_ColliderFrictionCoeff.Dispose();

            m_TetIndices.Dispose();
            m_Vertices.Dispose();
            m_Weights.Dispose();
            m_SurfIndices.Dispose();

            m_NeighbourSearchTuples.Dispose();
            m_NeighbourSortData.Dispose();
            m_GhostPartNeighbourCellIndex.Dispose();
            m_PartGhostNeighboursCount.Dispose();
            m_PartGhostNeighbours.Dispose();
            m_OrderNewToOld.Dispose();
            m_OrderOldToNew.Dispose();

            Simulation.OnBeginUpdate.RemoveListener(Priority, UpdateColliders);
            Simulation.OnPositionsEstimated.RemoveListener(Priority, DetectContactsAndPrestabilize);
            Simulation.OnParticlesRearranged.RemoveListener(Priority, RearrangeContactsAndGhostParts);
        }

        // ---- Constraint Events ----

        protected override void BindConstraintConstants(ComputeProgram program, bool velocities)
        {
            base.BindConstraintConstants(program, velocities);

            program.SetInt("NumGhostParts", m_NumGhostParts);
            program.SetInt("MaxGhostParts", m_MaxGhostParts);
            program.SetInt("NumColliders", m_Colliders.Count);
            program.SetFloat("ContactRadius", m_ContactRadius);
            program.SetFloat("ContactRadiusSqr", m_ContactRadius * m_ContactRadius);
            program.SetFloat("GhostNeighbourRadius", m_ContactRadius);
            program.SetFloat("GhostNeighbourRadiusSqr", m_ContactRadius * m_ContactRadius);
            program.SetFloat("InvGhostNeighbourRadius", 1 / m_ContactRadius);
            program.SetInt("MaxGhostNeighbours", m_MaxNeighbours);
            program.SetInt("MaxGhostNeighbourBuckets", m_MaxNeighbourBuckets);
        }

        protected override bool SolvePositionConstraint()
        {
            Simulation.BindGlobalConstants(Program);
            BindConstraintConstants(Program, false);
            SolveContacts(false, false);
            return true;
        }

        protected override bool SolveVelocityConstraint()
        {
            Simulation.BindGlobalConstants(Program);
            BindConstraintConstants(Program, true);
            SolveContacts(false, true);
            return true;
        }

        // ---- Collision Constraint Management ----

        private void SolveContacts(bool prestabilization, bool onVelocities)
        {
            if (!onVelocities)
            {
                Program.SetKernel("KerComputeDeltaOnPositions");
                Program.SetFrontBuffer("PartPositionR", prestabilization ? Simulation.Data.PositionOld : Simulation.Data.Position);
            }
            else
            {
                Program.SetKernel("KerComputeDeltaOnVelocities");
                Program.SetFrontBuffer("PartVelocityR", Simulation.Data.Velocity);
            }

            Program.SetFrontBuffer("PartInfoR", Simulation.Data.Info);
            Program.SetFrontBuffer("PartPositionOldR", Simulation.Data.PositionOld);
            Program.SetFrontBuffer("PartCtInfoR", m_PartContactInfo);
            Program.SetFrontBuffer("PartCtCountR", m_PartContactCount);
            Program.SetFrontBuffer("PartCtPointsNPlanesR", m_PartContactPointsNPlanes);
            Program.SetFrontBuffer("PartCtFrictionCoeffR", m_PartContactFrictionCoeff);
            Program.SetFrontBuffer("PartDeltaRW", Simulation.Data.Delta);
            Program.LaunchAuto(Simulation.NumParticles);
        }

        private void DetectContactsAndPrestabilize()
        {
            if (m_NumGhostParts == 0)
                return;

            Simulation.BindGlobalConstants(Program);
            BindConstraintConstants(Program, false);
            Simulation.SearchNeighbours();

            InterpolateVertices((float)Simulation.CurrentSubStep / Simulation.NumSubSteps);
            UpdateGhostParticles();
            SearchGhostPartNeighbours();
            DetectGhostParticlesContacts();

            for (var i = 0; i < m_PrestabIters; i++)
            {
                Simulation.ClearDeltas();
                SolveContacts(true, false);
                Simulation.ApplyDeltasOnPositions(TargetMask);
                Simulation.ApplyDeltasOnOldPositions(TargetMask);
            }
        }

        private void DetectGhostParticlesContacts()
        {
            Fill.Execute(m_PartContactCount.FrontBuffer, 0, 0, Simulation.NumParticles);

            BindConstraintConstants(Program, false);
            Simulation.BindPerGroupConstants(Program);
            Program.SetKernel("KerDetectGhostPartContacts");
            Program.SetFrontBuffer("PartInfoRW", Simulation.Data.Info);
            Program.SetFrontBuffer("PartPositionR", Simulation.Data.Position);
            Program.SetFrontBuffer("PartCtInfoRW", m_PartContactInfo);
            Program.SetFrontBuffer("PartCtCountRW", m_PartContactCount);
            Program.SetFrontBuffer("PartCtPointsNPlanesRW", m_PartContactPointsNPlanes);
            Program.SetFrontBuffer("PartCtFrictionCoeffRW", m_PartContactFrictionCoeff);
            Program.SetBuffer("PartGhostNeighboursCountR", m_PartGhostNeighboursCount);
            Program.SetBuffer("PartGhostNeighboursR", m_PartGhostNeighbours);
            Program.SetFrontBuffer("GhostPartPositionR", m_GhostPartPosition);
            Program.SetFrontBuffer("GhostPartPositionDeltaR", m_GhostPartPositionDelta);
            Program.SetFrontBuffer("GhostPartColliderIdR", m_GhostPartColliderId);
            Program.SetFrontBuffer("GhostPartTypeR", m_GhostPartType);
            Program.SetFrontBuffer("GhostPartClosestDistR", m_GhostPartClosestDist);
            Program.SetFrontBuffer("GhostPartClosestDirR", m_GhostPartClosestDir);
            Program.SetBuffer("ColliderFrictionCoeffR", m_ColliderFrictionCoeff);
            Program.LaunchAuto(Simulation.NumParticles);
        }

        private void UpdateGhostParticles()
        {
            if (m_NumGhostParts == 0)
                return;

            Program.SetKernel("KerUpdateGhostParts");
            Program.SetFrontBuffer("GhostPartPositionR", m_GhostPartPosition);
            Program.SetBackBuffer("GhostPartPositionRW", m_GhostPartPosition);
            Program.SetFrontBuffer("GhostPartPositionDeltaRW", m_GhostPartPositionDelta);
            Program.SetFrontBuffer("GhostPartTetIdR", m_GhostPartTetId);
            Program.SetFrontBuffer("GhostPartTriNTetBarycentricCoordsR", m_GhostPartTriNTetBarycentricCoords);
            Program.SetFrontBuffer("GhostPartClosestTriIdR", m_GhostPartClosestTriId);
            Program.SetFrontBuffer("GhostPartTypeRW", m_GhostPartType);
            Program.SetFrontBuffer("GhostPartClosestDistRW", m_GhostPartClosestDist);
            Program.SetFrontBuffer("GhostPartClosestDirRW", m_GhostPartClosestDir);
            Program.SetBuffer("VerticesR", m_Vertices);
            Program.SetBuffer("TetraIndicesR", m_TetIndices);
            Program.SetBuffer("SurfaceIndicesR", m_SurfIndices);
            Program.LaunchAuto(m_NumGhostParts);

            m_GhostPartPosition.SwapBuffers();
        }

        private void RearrangeContactsAndGhostParts()
        {
            if (m_NumGhostParts == 0)
                return;

            Simulation.BindGlobalConstants(Program);
            Program.SetKernel("KerRearrangeContacts");
            Simulation.BindParticleOrderBuffers(Program);
            Program.SetFrontBuffer("PartInfoR", Simulation.Data.Info);
            Program.SetFrontBuffer("PartCtInfoR", m_PartContactInfo);
            Program.SetFrontBuffer("PartCtCountR", m_PartContactCount);
            Program.SetFrontBuffer("PartCtPointsNPlanesR", m_PartContactPointsNPlanes);
            Program.SetFrontBuffer("PartCtFrictionCoeffR", m_PartContactFrictionCoeff);
            Program.SetBackBuffer("PartCtInfoRW", m_PartContactInfo);
            Program.SetBackBuffer("PartCtCountRW", m_PartContactCount);
            Program.SetBackBuffer("PartCtPointsNPlanesRW", m_PartContactPointsNPlanes);
            Program.SetBackBuffer("PartCtFrictionCoeffRW", m_PartContactFrictionCoeff);
            Program.LaunchAuto(Simulation.NumParticles);

            m_PartContactInfo.SwapBuffers();
            m_PartContactCount.SwapBuffers();
            m_PartContactPointsNPlanes.SwapBuffers();
            m_PartContactFrictionCoeff.SwapBuffers();
        }

        // ---- Collision Constraint Public Management ---

        public void AddCollider(SimulationPointCloudCollider collider)
        {
            m_Colliders.Add(collider);
        }

        public void RemoveCollider(SimulationPointCloudCollider collider)
        {
            m_Colliders.Remove(collider);
        }

        private void CreateGhostParticles()
        {
            m_NumGhostParts = 0;

            // Rasterize colliders into ghost particles, determine total no. of ghost particles.

            foreach (var collider in m_Colliders)
                m_NumGhostParts += collider.RasterizeCollider();

            // Emit the ghost particle, reallocate buffers and store their data.

            Assert.IsTrue(m_NumGhostParts <= m_MaxGhostParts, "Particle budget exceeded, unable to instantiate the ghost particles!");

            m_GhostPartPosition.Clear();
            m_GhostPartTriNTetBarycentricCoords.Clear();
            m_GhostPartClosestDir.Clear();
            m_GhostPartColliderId.Clear();
            m_GhostPartTetId.Clear();
            m_GhostPartClosestTriId.Clear();
            m_GhostPartType.Clear();
            m_GhostPartClosestDist.Clear();
            m_GhostPartIsSurface.Clear();

            m_ColliderFrictionCoeff.Clear();
            m_ColliderGhostPartStart.Clear();
            m_ColliderGhostPartCount.Clear();

            m_Vertices.Clear();
            m_TetIndices.Clear();
            m_SurfIndices.Clear();
            m_VerticesPrev.Clear();
            m_VerticesNext.Clear();

            m_GhostPartPosition.Resize(3 * m_MaxGhostParts);
            m_GhostPartPositionDelta.Resize(3 * m_MaxGhostParts);
            m_GhostPartTriNTetBarycentricCoords.Resize(7 * m_MaxGhostParts);
            m_GhostPartClosestDir.Resize(3 * m_MaxGhostParts);
            m_GhostPartColliderId.Reserve(m_NumGhostParts);
            m_GhostPartTetId.Reserve(m_NumGhostParts);
            m_GhostPartClosestTriId.Reserve(m_NumGhostParts);
            m_GhostPartType.Reserve(m_NumGhostParts);
            m_GhostPartClosestDist.Reserve(m_NumGhostParts);
            m_GhostPartIsSurface.Reserve(m_NumGhostParts);

            m_ColliderFrictionCoeff.Reserve(m_Colliders.Count);
            m_ColliderGhostPartStart.Reserve(m_Colliders.Count);
            m_ColliderGhostPartCount.Reserve(m_Colliders.Count);

            var colliderIndex = 0;
            var ghostParticleIndex = 0;

            foreach (var collider in m_Colliders)
            {
                collider.Id = colliderIndex;
                StoreColliderData(collider, ref ghostParticleIndex);
                ++colliderIndex;
            }

            // Perform initial copy of the vertex data for interpolation.

            for (var i = 0; i < m_Vertices.Count; ++i)
            {
                m_VerticesNext.Add(m_Vertices[i]);
                m_VerticesPrev.Add(m_Vertices[i]);
            }

            // Done, upload data to GPU.

            m_GhostPartPosition.UploadToBuffer();
            m_GhostPartColliderId.UploadToBuffer();
            m_GhostPartTetId.UploadToBuffer();
            m_GhostPartTriNTetBarycentricCoords.UploadToBuffer();
            m_GhostPartClosestTriId.UploadToBuffer();
            m_GhostPartType.UploadToBuffer();
            m_GhostPartClosestDist.UploadToBuffer();
            m_GhostPartClosestDir.UploadToBuffer();
            m_GhostPartIsSurface.UploadToBuffer();

            m_ColliderFrictionCoeff.UploadToBuffer();
            m_ColliderGhostPartStart.UploadToBuffer();
            m_ColliderGhostPartCount.UploadToBuffer();

            m_Vertices.UploadToBuffer();
            m_TetIndices.UploadToBuffer();
            m_SurfIndices.UploadToBuffer();
        }

        private void StoreColliderData(SimulationPointCloudCollider collider, ref int ghostParticleIndex)
        {
            // Upload Tetra Data.

            var verticesOffset = m_Vertices.Count;
            var tetIndicesOffset = m_TetIndices.Count;
            var surfIndicesOffset = m_SurfIndices.Count;

            var verticesSize = collider.Mesh.NumVertices;
            var tetIndicesSize = collider.Mesh.NumTetrahedrons * 4;
            var surfIndicesSize = collider.Mesh.NumTriangles * 3;

            m_Vertices.Reserve(verticesOffset + verticesSize);
            m_TetIndices.Reserve(tetIndicesOffset + tetIndicesSize);
            m_SurfIndices.Reserve(surfIndicesOffset + surfIndicesSize);
            m_Weights.Reserve(verticesOffset + verticesSize);

            // Vertices.

            for (int i = 0; i < verticesSize; ++i)
            {
                var weight = collider.Mesh.Weights[i];
                var vertex = collider.Mesh.Vertices[i];
                vertex = collider.transform.TransformPoint(vertex);
                m_Vertices.Add(vertex);
                m_Weights.Add(weight);
            }

            // Tetrahedron Indices. 

            for (int i = 0; i < collider.Mesh.NumTetrahedrons; ++i)
            {
                m_TetIndices.Add(collider.Mesh.Tetrahedrons[i].A + verticesOffset);
                m_TetIndices.Add(collider.Mesh.Tetrahedrons[i].B + verticesOffset);
                m_TetIndices.Add(collider.Mesh.Tetrahedrons[i].C + verticesOffset);
                m_TetIndices.Add(collider.Mesh.Tetrahedrons[i].D + verticesOffset);
            }

            // Surface Triangle Indices.

            for (int i = 0; i < collider.Mesh.NumTriangles; ++i)
            {
                m_SurfIndices.Add(collider.Mesh.Surface[i].A + verticesOffset);
                m_SurfIndices.Add(collider.Mesh.Surface[i].B + verticesOffset);
                m_SurfIndices.Add(collider.Mesh.Surface[i].C + verticesOffset);
            }

            // Ghost Particles Data.

            for (int i = 0; i < collider.NumGhostPart; ++i, ++ghostParticleIndex)
            {
                var type = collider.GhostPartClosestDists[i] > 0.0f ? 0             // 0 = Exterior
                         : collider.GhostPartClosestDists[i] > -ContactRadius ? 1   // 1 = Boundary
                         : 2;                                                       // 2 = Interior

                m_GhostPartPosition.WriteVector3(ghostParticleIndex, m_MaxGhostParts, collider.GhostPartPositions[i]);
                m_GhostPartTriNTetBarycentricCoords.WriteVector3(ghostParticleIndex, m_MaxGhostParts, collider.GhostPartClosestTriBarycentrics[i]);
                m_GhostPartTriNTetBarycentricCoords.WriteVector4(ghostParticleIndex + 3 * m_MaxGhostParts, m_MaxGhostParts, collider.GhostPartTetBarycentrics[i]);
                m_GhostPartClosestDir.WriteVector3(ghostParticleIndex, m_MaxGhostParts, collider.GhostPartClosestDirs[i]);
                m_GhostPartColliderId.Add(collider.Id);
                m_GhostPartTetId.Add(4 * collider.GhostPartTetIds[i] + tetIndicesOffset);
                m_GhostPartClosestTriId.Add(3 * collider.GhostPartClosestTriIds[i] + surfIndicesOffset);
                m_GhostPartClosestDist.Add(collider.GhostPartClosestDists[i]);
                m_GhostPartType.Add(type);
                m_GhostPartIsSurface.Add(collider.GhostPartIsSurface[i]);
            }

            // Collider Data.

            m_ColliderGhostPartCount.Add(collider.NumGhostPart);
            m_ColliderFrictionCoeff.Add(collider.FrictionCoefficient);
        }

        private void UpdateColliders()
        {
            int vertexIdx = 0;

            // Swap prev and next vertices.
            MathUtility.Swap(ref m_VerticesNext, ref m_VerticesPrev);

            foreach (var collider in Colliders)
            {
                vertexIdx = UpdateVertices(vertexIdx, collider);
                m_ColliderFrictionCoeff[collider.Id] = collider.FrictionCoefficient;
            }

            m_VerticesNext.UploadToBuffer();
            m_ColliderFrictionCoeff.UploadToBuffer();
        }

        private int UpdateVertices(int vertexIdx, SimulationPointCloudCollider collider)
        {
            for (int i = 0; i < collider.Mesh.NumVertices; ++i, ++vertexIdx)
            {
                var vertex = collider.Mesh.Vertices[i];
                vertex = collider.transform.TransformPoint(vertex);
                m_VerticesNext[vertexIdx] = vertex;
            }

            return vertexIdx;
        }

        private void SearchGhostPartNeighbours()
        {
            Simulation.BindGlobalConstants(Program);
            BindConstraintConstants(Program, false);

            Program.SetKernel("KerOrderByNeighbourhood");
            Program.SetFrontBuffer("GhostPartPositionR", m_GhostPartPosition);
            Program.SetBackBuffer("GhostNeighbourSearchTuplesRW", m_NeighbourSearchTuples);
            Program.LaunchAuto(m_NumGhostParts);

            CountSort.Execute(m_NeighbourSortData,
                m_NeighbourSearchTuples.BackBuffer,
                m_NeighbourSearchTuples.FrontBuffer,
                m_NumGhostParts,
                m_MaxNeighbourBuckets
            );

            Fill.Execute(m_OrderOldToNew, ParticleConstants.Null, 0, m_NumGhostParts);
            Fill.Execute(m_OrderNewToOld, ParticleConstants.Null, 0, m_NumGhostParts);

            Program.SetKernel("KerRearrangeParticlesA");
            Program.SetFrontBuffer("GhostNeighbourSearchTuplesR", m_NeighbourSearchTuples);
            Program.SetFrontBuffer("GhostPartPositionR", m_GhostPartPosition);
            Program.SetFrontBuffer("GhostPartPositionDeltaR", m_GhostPartPositionDelta);
            Program.SetFrontBuffer("GhostPartColliderIdR", m_GhostPartColliderId);
            Program.SetFrontBuffer("GhostPartTypeR", m_GhostPartType);
            Program.SetFrontBuffer("GhostPartClosestDistR", m_GhostPartClosestDist);
            Program.SetFrontBuffer("GhostPartClosestDirR", m_GhostPartClosestDir);
            Program.SetBackBuffer("GhostPartPositionRW", m_GhostPartPosition);
            Program.SetBackBuffer("GhostPartPositionDeltaRW", m_GhostPartPositionDelta);
            Program.SetBackBuffer("GhostPartColliderIdRW", m_GhostPartColliderId);
            Program.SetBackBuffer("GhostPartTypeRW", m_GhostPartType);
            Program.SetBackBuffer("GhostPartClosestDistRW", m_GhostPartClosestDist);
            Program.SetBackBuffer("GhostPartClosestDirRW", m_GhostPartClosestDir);
            Program.LaunchAuto(m_NumGhostParts);

            m_GhostPartPosition.SwapBuffers();
            m_GhostPartPositionDelta.SwapBuffers();
            m_GhostPartColliderId.SwapBuffers();
            m_GhostPartType.SwapBuffers();
            m_GhostPartClosestDist.SwapBuffers();
            m_GhostPartClosestDir.SwapBuffers();

            Program.SetKernel("KerRearrangeParticlesB");
            Program.SetFrontBuffer("GhostPartTetIdR", m_GhostPartTetId);
            Program.SetFrontBuffer("GhostNeighbourSearchTuplesR", m_NeighbourSearchTuples);
            Program.SetFrontBuffer("GhostPartClosestTriIdR", m_GhostPartClosestTriId);
            Program.SetFrontBuffer("GhostPartTriNTetBarycentricCoordsR", m_GhostPartTriNTetBarycentricCoords);
            Program.SetFrontBuffer("GhostPartIsSurfaceR", m_GhostPartIsSurface);
            Program.SetBackBuffer("GhostPartTetIdRW", m_GhostPartTetId);
            Program.SetBackBuffer("GhostPartClosestTriIdRW", m_GhostPartClosestTriId);
            Program.SetBackBuffer("GhostPartTriNTetBarycentricCoordsRW", m_GhostPartTriNTetBarycentricCoords);
            Program.SetBackBuffer("GhostPartIsSurfaceRW", m_GhostPartIsSurface);
            Program.SetBuffer("GhostPartOrderOldToNewRW", m_OrderOldToNew);
            Program.SetBuffer("GhostPartOrderNewToOldRW", m_OrderNewToOld);
            Program.SetBuffer("GhostPartNeighbourCellIndexRW", m_GhostPartNeighbourCellIndex);
            Program.LaunchAuto(m_NumGhostParts);

            m_GhostPartTetId.SwapBuffers();
            m_GhostPartClosestTriId.SwapBuffers();
            m_GhostPartTriNTetBarycentricCoords.SwapBuffers();
            m_GhostPartIsSurface.SwapBuffers();

            Program.SetKernel("KerFindGhostNeighbours");
            Program.SetFrontBuffer("PartPositionR", Simulation.Data.Position);
            Program.SetFrontBuffer("GhostPartPositionR", m_GhostPartPosition);
            Program.SetBuffer("PartGhostNeighboursCountRW", m_PartGhostNeighboursCount);
            Program.SetBuffer("PartGhostNeighboursRW", m_PartGhostNeighbours);
            Program.SetBuffer("GhostNeighbourSearchOffsetR", m_NeighbourSortData.KeyOffsBuffer);
            Program.SetBuffer("GhostNeighbourSearchCountR", m_NeighbourSortData.KeyHistBuffer);
            Program.LaunchAuto(Simulation.NumParticles);
        }

        private void InterpolateVertices(float t)
        {
            Program.SetKernel("KerInterpolateVertices");
            Program.SetInt("NumVertices", m_Vertices.Count);
            Program.SetFloat("InterpolationWeight", t);
            Program.SetBuffer("VerticesRW", m_Vertices.Buffer);
            Program.SetBuffer("VerticesPrevR", m_VerticesPrev.Buffer);
            Program.SetBuffer("VerticesNextR", m_VerticesNext.Buffer);
            Program.LaunchAuto(m_Vertices.Count);
        }
    }
}