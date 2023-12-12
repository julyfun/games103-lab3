using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

class Triangle
{
    public Triangle(Vector3[] x, Vector3[] v, int[] index, int belong)
    {
        this.x = x;
        this.v = v;
        this.index = index;
        this.belong = belong;
    }
    public Vector3[] x;
    public Vector3[] v;
    public int[] index;
    public int belong;
}

class NewtonSolver
{
    int iteration_num = 20;
    float solve(FuncC1 fun, float init_x)
    {
        float x = init_x;
        for (int i = 0; i < this.iteration_num; i++)
        {
            x = x - fun.value(x) / fun.derivative(x);
        }
        return x;
    }

}

interface FuncC1
{
    float derivative(float x);
    float value(float x);
}

class Polynomial : FuncC1
{
    float[] a;
    Polynomial(float[] a)
    {
        this.a = a;
    }
    public float derivative(float x)
    {
        var sum = 0f;
        for (int i = 1; i < a.Length; i++)
        {
            sum += a[i] * i * Mathf.Pow(x, i - 1);
        }
        return sum;
    }
    public float value(float x)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * Mathf.Pow(x, i);
        }
        return sum;
    }
}

class NodeTriangleCollision
{
    Vector3[] x;
    Vector3[] v;

    /*
    - Applying the Rules:

    Combining the product rule and the chain rule, we get:

    df/dt = d/dt [x0 dot (x1 cross x2)]

    = x0'(t) dot (x1 cross x2) + x0 dot (d/dt [(x1 cross x2)])

    - Expanding the Terms:

    Next, we need to differentiate the cross product term:

    d/dt [(x1 cross x2)] = (x1' cross x2) + (x1 cross x2')
    */

}

public class FvmCollider : MonoBehaviour
{
    // Start is called before the first frame update
    public int collide_tree_min_size = 8;

    void Start()
    {

    }

    bool node_inside_cube(Vector3 node, Vector3 small, Vector3 big)
    {
        return small.x <= node.x && node.x <= big.x
            && small.y <= node.y && node.y <= big.y
            && small.z <= node.z && node.z <= big.z;
    }

    Vector3 tri_normal_012(Vector3[] x)
    {
        Vector3 x10 = x[1] - x[0];
        Vector3 x20 = x[2] - x[0];
        return Vector3.Cross(x10, x20).normalized;
    }

    bool node_inside_tri(Vector3[] x, Vector3 p)
    {
        var normal_012 = this.tri_normal_012(x);
        for (int i = 0; i < 2; i++)
        {
            int nxt = (i + 1) % 3;
            if (Vector3.Dot(Vector3.Cross(x[i] - p, x[nxt] - p), normal_012) <= 0)
            {
                return false;
            }
        }
        return true;
    }

    bool tri_seg_collide(Vector3[] x, Vector3 xa, Vector3 xb)
    {
        Vector3 x0a = x[0] - xa;
        Vector3 xba = xb - xa;
        Vector3 x10 = x[1] - x[0];
        Vector3 x20 = x[2] - x[0];
        float t = Vector3.Dot(x0a, Vector3.Cross(x10, x20)) / Vector3.Dot(xba, Vector3.Cross(x10, x20));
        if (t < 0 || t > 1)
        {
            return false;
        }
        Vector3 on_plane = xa * (1 - t) + xb * t;
        return this.node_inside_tri(x, on_plane);
    }

    // suppose x1 < x2, ...
    void collide_split(List<Triangle> triangles, Vector3 xyz1, Vector3 xyz2)
    {
        float x1 = xyz1.x, y1 = xyz1.y, z1 = xyz1.z;
        float x2 = xyz2.x, y2 = xyz2.y, z2 = xyz2.z;
        var x_mid = (x1 + x2) / 2;
        var y_mid = (y1 + y2) / 2;
        var z_mid = (z1 + z2) / 2;
        // 使用递归八分法对所有三角形进行碰撞检测
        if (triangles.Count == 0) return;
        if (triangles.Count <= this.collide_tree_min_size)
        {
            for (int i = 0; i < triangles.Count; i++)
            {
                for (int j = i + 1; j < triangles.Count; j++)
                {
                    if (tri_collide(triangles[i], triangles[j]).collide)
                    {
                        // tri_collide_response();
                    }
                }
            }
            return;
        }
        var on_line_triangles = new List<Triangle>();
        var off_line_triangles = new List<Triangle>();
        // 有三条线段需要判断 
        var three_mid_segs = new Vector3[3, 2] {
            { new Vector3(x1, y_mid, z_mid), new Vector3(x2, y_mid, z_mid )},
            { new Vector3(x_mid, y1, z_mid), new Vector3(x_mid, y2, z_mid )},
            { new Vector3(x_mid, y_mid, z1), new Vector3(x_mid, y_mid, z2 )},
        };

        for (int i = 0; i < triangles.Count; i++)
        {
            bool collide_with_a_line = false;
            for (int j = 0; j < 3; j++)
            {
                if (this.tri_seg_collide(triangles[i].x, three_mid_segs[j, 0], three_mid_segs[j, 1]))
                {
                    collide_with_a_line = true;
                    break;
                }
            }
            if (collide_with_a_line)
            {
                on_line_triangles.Add(triangles[i]);
            }
            else
            {
                off_line_triangles.Add(triangles[i]);
            }
        }

        for (int i = 0; i < on_line_triangles.Count; i++)
        {
            for (int j = 0; j < off_line_triangles.Count; i++)
            {
                if (this.tri_collide(on_line_triangles[i], off_line_triangles[j]).collide)
                {
                    // tri_collide_response();
                }
            }
        }
        var triangles_group = new List<Triangle>[8];
        // var group_bounds = new Vector3[8, 2] {
        //      { new Vector3(x1, y1, z1), new Vector3(x_mid, y_mid, z_mid)},
        // };
        var group_bounds = new Vector3[8, 2] {
            { new Vector3(x1, y1, z1), new Vector3(x_mid, y_mid, z_mid) }, // Top-front left octant
            { new Vector3(x_mid, y1, z1), new Vector3(x2, y_mid, z_mid) }, // Top-front right octant
            { new Vector3(x1, y_mid, z1), new Vector3(x_mid, y2, z_mid) }, // Top-back left octant
            { new Vector3(x_mid, y_mid, z1), new Vector3(x2, y2, z_mid) }, // Top-back right octant
            { new Vector3(x1, y1, z_mid), new Vector3(x_mid, y_mid, z2) }, // Bottom-front left octant
            { new Vector3(x_mid, y1, z_mid), new Vector3(x2, y_mid, z2) }, // Bottom-front right octant
            { new Vector3(x1, y_mid, z_mid), new Vector3(x_mid, y2, z2) }, // Bottom-back left octant
            { new Vector3(x_mid, y_mid, z_mid), new Vector3(x2, y2, z2) }, // Bottom-back right octant
        };
        // offline 只需要判断其中一个点的位置就可以了
        for (int i = 0; i < off_line_triangles.Count; i++)
        {
            for (int j = 0; j < 8; j++)
            {
                if (this.node_inside_cube(off_line_triangles[i].x[0], group_bounds[j, 0], group_bounds[j, 1]))
                {
                    triangles_group[j].Add(off_line_triangles[i]);
                    break;
                }
            }
        }
        for (int i = 0; i < 8; i++)
        {
            this.collide_split(triangles_group[i], group_bounds[i, 0], group_bounds[i, 1]);
        }

    }

    // 这里 float 是求得的解
    (bool collide, float t) tri_collide(Triangle t1, Triangle t2)
    {
        if (t1.belong == t2.belong)
        {
            return (false, 0);
        }
        return (false, 0);
    }
    void tri_collide_response()
    {
    }

    // Update is called once per frame
    void Update()
    {
        // 获取所有的 fvm
        FVM[] objects = GameObject.FindObjectsOfType<FVM>();
        var triangles = new List<Triangle>();
        for (int i = 0; i < objects.Length; i++)
        {
            Vector3[] real_x = objects[i].get_real_x();
            Vector3[] real_v = objects[i].get_real_v();
            TriIndex[] those_tri_index = objects[i].get_tri_index();
            for (int j = 0; j < real_x.Length; j++)
            {
                triangles.Add(new Triangle(
                    new Vector3[]{
                        real_x[those_tri_index[j].index[0]],
                        real_x[those_tri_index[j].index[1]],
                        real_x[those_tri_index[j].index[2]],
                    },
                    new Vector3[]{
                        real_v[those_tri_index[j].index[0]],
                        real_v[those_tri_index[j].index[1]],
                        real_v[those_tri_index[j].index[2]],
                    },
                    those_tri_index[j].index,
                    i
                ));
            }
        }
        this.collide_split(triangles, new Vector3(-10, -10, 10), new Vector3(10, 10, 10));
    }
}
