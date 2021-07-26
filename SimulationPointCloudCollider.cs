using Particleworks.Geometry;
using Particleworks.Math;
using Particleworks.Simulation;
using Particleworks.Utils;
using System.Collections.Generic;
using Unity.UNetWeaver;
using UnityEngine;
using GeometryUtility = Particleworks.Geometry.GeometryUtility;

namespace Particleworks.Collisions
{
    ////////////////////////////////////////////////////////////////////////////////
    /// <summary>
    /// 
    /// Object meshed with tetrahedra to fullfill with ghost particles.
    ///
    /// </summary>
    ////////////////////////////////////////////////////////////////////////////////

    public class SimulationPointCloudCollider : MonoBehaviour
    {
        //== Properties ============================================================

        public PointCloudCollisionConstraint Target { get => m_Target; }

        public float FrictionCoefficient => m_FrictionCoefficient;

        public int Id { get; set; }

        public int NumGhostPart { get; private set; }

        public float NarrowBand => m_NarrowBand;

        public IReadOnlyList<Vector3> GhostPartPositions { get => m_GhostPartPositions; }

        public IList<float> GhostPartClosestDists { get => m_GhostPartClosestDists; }

        public IReadOnlyList<float> GhostPartSign { get => m_GhostPartSign; }

        public IList<Vector3> GhostPartClosestDirs { get => m_GhostPartClosestDirs; }

        public IReadOnlyList<int> GhostPartClosestTriIds { get => m_GhostPartClosestTriIds; }

        public IReadOnlyList<Vector3> GhostPartClosestTriBarycentrics { get => m_GhostPartClosestTriBarycentrics; }

        public IReadOnlyList<int> GhostPartTetIds { get => m_GhostPartTetIds; }

        public IReadOnlyList<Vector4> GhostPartTetBarycentrics { get => m_GhostPartTetBarycentrics; }

        public IReadOnlyList<int> GhostPartIsSurface { get => m_GhostPartIsSurface; }

        //== Members ===============================================================

        // ---- Params ----

        [SerializeField]
        [RuntimeReadOnly]
        private PointCloudCollisionConstraint m_Target;

        [SerializeField]
        [RuntimeReadOnly]
        private float m_NarrowBand;

        [SerializeField]
        private float m_FrictionCoefficient;

        [RuntimeReadOnly]
        public TetrahedronMesh Mesh;

        // ---- Private ----

        private readonly List<Vector3> m_GhostPartPositions = new List<Vector3>();

        private readonly List<float> m_GhostPartSign = new List<float>();

        private readonly List<float> m_GhostPartClosestDists = new List<float>();

        private readonly List<Vector3> m_GhostPartClosestDirs = new List<Vector3>();

        private readonly List<int> m_GhostPartClosestTriIds = new List<int>();

        private readonly List<Vector3> m_GhostPartClosestTriBarycentrics = new List<Vector3>();

        private readonly List<int> m_GhostPartTetIds = new List<int>();

        private readonly List<Vector4> m_GhostPartTetBarycentrics = new List<Vector4>();

        private readonly List<int> m_GhostPartIsSurface = new List<int>();

        //== Methods ===============================================================

        // ---- Unity events ----

        private void OnEnable()
        {
            if (Target == null)
                return;

            AddToConstraint();
        }

        private void OnDisable()
        {
            if (Target == null)
                return;

            RemoveFromConstraint();
        }

        // ---- Collider management ----

        public int RasterizeCollider()
        {
            // Initialize ghost particles state.

            m_GhostPartPositions.Clear();
            m_GhostPartSign.Clear();
            m_GhostPartTetBarycentrics.Clear();
            m_GhostPartTetIds.Clear();
            m_GhostPartClosestTriIds.Clear();
            m_GhostPartClosestTriBarycentrics.Clear();
            m_GhostPartClosestDists.Clear();
            m_GhostPartClosestDirs.Clear();
            m_GhostPartIsSurface.Clear();

            // Determine amount of particles to create.

            NumGhostPart = 0;

            // Rasterization needs to happen on world coordinates. Fetch the current transform
            // matrices to go back and forth from world to local.

            var transform = this.transform.localToWorldMatrix;
            var invTransform = this.transform.worldToLocalMatrix;
            var voxelSize = Target.PartAvgeSpacing;

            // Determine AABB of mesh in world-space.

            var minBounds = Vector3.positiveInfinity;
            var maxBounds = Vector3.negativeInfinity;

            for (int i = 0; i < Mesh.NumVertices; ++i)
            {
                var vertex = transform.MultiplyPoint3x4(Mesh.Vertices[i]);
                minBounds = Vector3.Min(minBounds, vertex);
                maxBounds = Vector3.Max(maxBounds, vertex);
            }

            minBounds -= Vector3.one * NarrowBand;
            maxBounds += Vector3.one * NarrowBand;

            // Determine grid coordinates and size.

            var boundsSize = maxBounds - minBounds;
            var gridSize = MathUtility.FloorToInt(boundsSize / voxelSize) + IntVector3.one;

            // Allocate rasterization buffers.

            var occupiedBuffer = new bool[gridSize.x, gridSize.y, gridSize.z];
            var insideBuffer = new bool[gridSize.x, gridSize.y, gridSize.z];
            var distanceBuffer = new float[gridSize.x, gridSize.y, gridSize.z];
            var tetIdBuffer = new int[gridSize.x, gridSize.y, gridSize.z];
            var tetBarCoordsBuffer = new Vector4[gridSize.x, gridSize.y, gridSize.z];

            // For each tetrahedron...

            for (int i = 0; i < Mesh.NumTetrahedrons; ++i)
            {
                // Compute world-space vertex coordinates.

                var tetrahedron = Mesh.Tetrahedrons[i];
                var vertexA = transform.MultiplyPoint3x4(Mesh.Vertices[tetrahedron.A]);
                var vertexB = transform.MultiplyPoint3x4(Mesh.Vertices[tetrahedron.B]);
                var vertexC = transform.MultiplyPoint3x4(Mesh.Vertices[tetrahedron.C]);
                var vertexD = transform.MultiplyPoint3x4(Mesh.Vertices[tetrahedron.D]);

                // Find tetra AABB and grid coordinates.

                var minTetraBounds = Vector3.Min(vertexA, Vector3.Min(vertexB, Vector3.Min(vertexC, vertexD))) - m_NarrowBand * Vector3.one;
                var maxTetraBounds = Vector3.Max(vertexA, Vector3.Max(vertexB, Vector3.Max(vertexC, vertexD))) + m_NarrowBand * Vector3.one;

                var minTetraCoord = MathUtility.FloorToInt((minTetraBounds - minBounds) / voxelSize);
                var maxTetraCoord = MathUtility.CeilToInt((maxTetraBounds - minBounds) / voxelSize);

                // Determine occupancy.

                for (int z = minTetraCoord.z; z <= maxTetraCoord.z; z++)
                    for (int y = minTetraCoord.y; y <= maxTetraCoord.y; y++)
                        for (int x = minTetraCoord.x; x <= maxTetraCoord.x; x++)
                        {
                            // Compute world-space cell center position and check if it lies within the current tetrahedron.

                            var coord = new IntVector3(x, y, z);
                            var p = minBounds + (coord * voxelSize) + 0.5f * voxelSize * Vector3.one;

                            var isInsideTetra = ShapeUtility.PtInsideTetra(p, vertexA, vertexB, vertexC, vertexD, out Vector4 tetBarycentricCoords);
                            var distanceToTetra = isInsideTetra ? 0.0f : Vector3.Distance(p, GeometryUtility.ClosestPtToTetra(p, vertexA, vertexB, vertexC, vertexD));

                            if (distanceToTetra <= m_NarrowBand)
                            {
                                // Flag as occupied, and replace previous values if this primitive is closer than the previous one.

                                if (occupiedBuffer[x, y, z] == false || distanceToTetra < distanceBuffer[x, y, z])
                                {
                                    occupiedBuffer[x, y, z] = true;
                                    distanceBuffer[x, y, z] = distanceToTetra;
                                    insideBuffer[x, y, z] = isInsideTetra;
                                    tetIdBuffer[x, y, z] = i;
                                    tetBarCoordsBuffer[x, y, z] = tetBarycentricCoords;
                                }
                            }
                        }
            }

            // Once occupancy has been determined for all voxels, iterate through them and generate
            // the corresponding neighborhood data.

            for (int z = 0; z < gridSize.z; z++)
                for (int y = 0; y < gridSize.y; y++)
                    for (int x = 0; x < gridSize.x; x++)
                    {
                        if (occupiedBuffer[x, y, z] == false) continue;

                        // Compute world-space cell center position. Determine closest surface triangle in local-space and
                        // its corresponding projected barycentric coordinates.

                        var coord = new IntVector3(x, y, z);
                        var p = minBounds + (coord * voxelSize) + 0.5f * voxelSize * Vector3.one;
                        var q = invTransform.MultiplyPoint3x4(p);

                        GeometryUtility.ClosestSurfaceTriangle(q, Mesh.Surface, Mesh.Vertices,
                                                               out int closestTriId,
                                                               out Vector3 closestTriBarycentricCoords);

                        // Compute projected point in world space and determine distance and direction.

                        var triangle = Mesh.Surface[closestTriId];
                        var triVertexA = transform.MultiplyPoint3x4(Mesh.Vertices[triangle.A]);
                        var triVertexB = transform.MultiplyPoint3x4(Mesh.Vertices[triangle.B]);
                        var triVertexC = transform.MultiplyPoint3x4(Mesh.Vertices[triangle.C]);
                        var pointInSurface = closestTriBarycentricCoords[0] * triVertexA
                                            + closestTriBarycentricCoords[1] * triVertexB
                                            + closestTriBarycentricCoords[2] * triVertexC;

                        var closestDir = p - pointInSurface;
                        var closestDist = closestDir.magnitude;
                        closestDir /= 1e-10f + closestDist;
                        var sign = insideBuffer[x, y, z] ? -1.0f : 1.0f;
                        var isSurface = closestDist < voxelSize ? 1 : 0;

                        // Finally, store ghost particle data.

                        m_GhostPartPositions.Add(p);
                        m_GhostPartSign.Add(sign);
                        m_GhostPartTetBarycentrics.Add(tetBarCoordsBuffer[x, y, z]);
                        m_GhostPartTetIds.Add(tetIdBuffer[x, y, z]);
                        m_GhostPartClosestTriIds.Add(closestTriId);
                        m_GhostPartClosestTriBarycentrics.Add(closestTriBarycentricCoords);
                        m_GhostPartClosestDists.Add(sign * closestDist);
                        m_GhostPartClosestDirs.Add(sign * closestDir);
                        m_GhostPartIsSurface.Add(isSurface);

                        // Done, proceed onto next!

                        ++NumGhostPart;
                    }

            // Return no. of ghost generated particles.

            return NumGhostPart;
        }

        protected void AddToConstraint()
        {
            Target.AddCollider(this);
        }

        protected void RemoveFromConstraint()
        {
            Target.RemoveCollider(this);
        }

        protected virtual void OnDrawGizmos()
        {
            Gizmos.color = Color.green;
            Gizmos.matrix = Matrix4x4.identity;

            if (Mesh != null)
            {
                for (int i = 0; i < Mesh.NumTetrahedrons; ++i)
                {
                    var tetrahedron = Mesh.Tetrahedrons[i];
                    var vertexA = transform.TransformPoint(Mesh.Vertices[tetrahedron.A]);
                    var vertexB = transform.TransformPoint(Mesh.Vertices[tetrahedron.B]);
                    var vertexC = transform.TransformPoint(Mesh.Vertices[tetrahedron.C]);
                    var vertexD = transform.TransformPoint(Mesh.Vertices[tetrahedron.D]);

                    Gizmos.DrawLine(vertexA, vertexB);
                    Gizmos.DrawLine(vertexB, vertexC);
                    Gizmos.DrawLine(vertexC, vertexA);
                    Gizmos.DrawLine(vertexA, vertexD);
                    Gizmos.DrawLine(vertexB, vertexD);
                    Gizmos.DrawLine(vertexC, vertexD);
                }
            }
        }
    }
}