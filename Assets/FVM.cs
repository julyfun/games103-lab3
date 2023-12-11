using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;
using UnityEngine.UIElements;
using System.Linq.Expressions;
using Unity.VisualScripting;
using System.Runtime.InteropServices;
using System.Xml.Serialization;

public class FVM : MonoBehaviour
{
    [Range(0.0f, 1.0f)]
    public float col_restitution = 0.5f;
    [Range(0.00166f, 0.002222f)]
    public float dt = 0.00166f;
    [ShowOnly] public float damp = 0.999f;
    [Range(1, 10)]
    public int calculation_per_frame = 3;
    [Range(0.05f, 20.0f)]
    public float mass = 1;
    [Range(0f, 30000f)]
    public float stiffness_0 = 20000.0f;
    [Range(0f, 8000f)]
    public float stiffness_1 = 5000.0f;
    [Range(0, 5000f)]
    public float stiffness_2 = 2000.0f;
    [Range(0, 50f)]
    public float stiffness_3 = 10.0f;
    public bool use_hyper = false;
    [Range(0, 4)]
    public int hyper_model = 0;
    [ShowOnly] public String hyper_model_name;
    private Dictionary<int, String> hyper_model_name_map = new Dictionary<int, String>()
    {
        {0, "StVK"},
        {1, "NeoHookean"},
        {2, "MooneyRivlin1"},
        {3, "MooneyRivlin-Wiki"},
        {4, "Fung"},
    };
    public int selected_x_id = -1;
    public float z_when_selected = 0f;
    public bool mouse_pressed = false;
    public bool use_superficial_collision = true;
    public float average_edge_length = 0f;
    float radius = 0f;

    int[] Tet;
    int tet_number;         //The number of tetrahedra

    Vector3[] outer_force;
    Vector3[] V;
    Vector3[] X;
    int num_nodes;             //The number of vertices

    Matrix4x4[] inv_dm;

    //For Laplacian smoothing.
    Vector3[] V_sum;
    int[] ref_count;

    SVD svd = new SVD();
    bool debug = false;

    // Start is called before the first frame update
    void Start()
    {
        this.GetComponent<LineRenderer>().enabled = false;
        // FILO IO: Read the house model from files.
        // The model is from Jonathan Schewchuk's Stellar lib.
        {
            // 存储每个四面体四个点的标号
            string fileContent = File.ReadAllText("Assets/house2.ele");
            string[] Strings = fileContent.Split(new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);

            tet_number = int.Parse(Strings[0]);
            Tet = new int[tet_number * 4];

            for (int tet = 0; tet < tet_number; tet++)
            {
                Tet[tet * 4 + 0] = int.Parse(Strings[tet * 5 + 4]) - 1;
                Tet[tet * 4 + 1] = int.Parse(Strings[tet * 5 + 5]) - 1;
                Tet[tet * 4 + 2] = int.Parse(Strings[tet * 5 + 6]) - 1;
                Tet[tet * 4 + 3] = int.Parse(Strings[tet * 5 + 7]) - 1;
            }
        }
        {
            // 读取每个标号的点的坐标
            string fileContent = File.ReadAllText("Assets/house2.node");
            string[] Strings = fileContent.Split(new char[] { ' ', '\t', '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries);
            num_nodes = int.Parse(Strings[0]);
            X = new Vector3[num_nodes];
            for (int i = 0; i < num_nodes; i++)
            {
                X[i].x = float.Parse(Strings[i * 5 + 5]) * 0.4f;
                X[i].y = float.Parse(Strings[i * 5 + 6]) * 0.4f;
                X[i].z = float.Parse(Strings[i * 5 + 7]) * 0.4f;
            }
            //Centralize the model.
            // 若 mass 一样，则重心为 0, 0, 0 (物体坐标系下)
            Vector3 center = Vector3.zero;
            for (int i = 0; i < num_nodes; i++) center += X[i];
            center = center / num_nodes;
            for (int i = 0; i < num_nodes; i++)
            {
                X[i] -= center;
                float temp = X[i].y;
                X[i].y = X[i].z;
                X[i].z = temp;
            }
        }
        // 仅供测试
        /*tet_number=1;
        Tet = new int[tet_number*4];
        Tet[0]=0;
        Tet[1]=1;
        Tet[2]=2;
        Tet[3]=3;

        number=4;
        X = new Vector3[number];
        V = new Vector3[number];
        Force = new Vector3[number];
        X[0]= new Vector3(0, 0, 0);
        X[1]= new Vector3(1, 0, 0);
        X[2]= new Vector3(0, 1, 0);
        X[3]= new Vector3(0, 0, 1);*/


        //Create triangle mesh.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];

            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }

        int[] triangles = new int[tet_number * 12];
        for (int t = 0; t < tet_number * 4; t++)
        {
            triangles[t * 3 + 0] = t * 3 + 0;
            triangles[t * 3 + 1] = t * 3 + 1;
            triangles[t * 3 + 2] = t * 3 + 2;
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        // vertices 中一个坐标出现了好多次..
        mesh.vertices = vertices;
        mesh.triangles = triangles;
        mesh.RecalculateNormals();


        V = new Vector3[num_nodes];
        outer_force = new Vector3[num_nodes];
        V_sum = new Vector3[num_nodes];
        ref_count = new int[num_nodes];
        for (int i = 0; i < tet_number; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                // 四面体四个点均贡献到其它点
                this.ref_count[this.Tet[i * 4 + j]] += 4;
            }
        }

        // 计算半径
        for (int i = 0; i < num_nodes; i++)
        {
            float dis = Vector3.Distance(X[i], Vector3.zero);
            if (dis > this.radius)
            {
                this.radius = dis;
            }
        }
        // calculate average edge length
        for (int i = 0; i < tet_number; i++)
        {
            for (int j = 0; j < 3; j++)
                for (int k = j + 1; k < 4; k++)
                {
                    int v1 = this.Tet[i * 4 + j];
                    int v2 = this.Tet[i * 4 + k];
                    float dis = Vector3.Distance(X[v1], X[v2]);
                    this.average_edge_length += dis;
                }
        }
        this.average_edge_length /= tet_number * 6;

        // allocate and assign inv_Dm
        this.inv_dm = new Matrix4x4[tet_number];
        for (int i = 0; i < tet_number; i++)
        {
            this.inv_dm[i] = Build_Edge_Matrix(i).inverse;
        }
    }

    Matrix4x4 Build_Edge_Matrix(int tet)
    {
        Matrix4x4 ret = Matrix4x4.zero;
        ret[3, 3] = 1f;         // 求逆应该需要这个
        for (int i = 0; i < 3; i++)
        {
            // 定义平衡位置
            ret.SetColumn(i,
                this.X[this.Tet[tet * 4 + i + 1]] - this.X[this.Tet[tet * 4 + 0]]);
        }
        return ret;
    }


    Matrix4x4 get_first_pk_stress1(Matrix4x4 deformation_gradient)
    {
        var green_strain =
            Mats.dot(
                Mats.minus(deformation_gradient.transpose * deformation_gradient, Matrix4x4.identity),
                0.5f
            );
        var second_pk_stress =
            Mats.add(
                Mats.dot(2f * this.stiffness_1, green_strain),
                Mats.dot(
                    this.stiffness_0 * Mats.trace(green_strain),
                    Matrix4x4.identity
                )
            );
        var first_pk_stress = deformation_gradient * second_pk_stress;
        return first_pk_stress;
    }

    Matrix4x4 get_stress_tensor_stvk(float lambda1, float lambda2, float lambda3)
    {
        // 20000, 5000
        float i = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
        float d1 = this.stiffness_0 * (i - 3) * lambda1 / 2
            + this.stiffness_1 * (lambda1 * lambda1 * lambda1 - lambda1);
        float d2 = this.stiffness_0 * (i - 3) * lambda2 / 2
            + this.stiffness_1 * (lambda2 * lambda2 * lambda2 - lambda2);
        float d3 = this.stiffness_0 * (i - 3) * lambda3 / 2
            + this.stiffness_1 * (lambda3 * lambda3 * lambda3 - lambda3);
        Matrix4x4 ret = Matrix4x4.zero;
        ret[0, 0] = d1;
        ret[1, 1] = d2;
        ret[2, 2] = d3;
        ret[3, 3] = 1f;
        return ret;
    }

    Matrix4x4 get_stress_tensor_neo_hookean(float lambda1, float lambda2, float lambda3)
    {
        // 20000, 5000
        //https://www.cs.toronto.edu/~jacobson/seminar/sifakis-course-notes-2012.pdf
        float[] l = { lambda1, lambda2, lambda3 };
        float i3 = lambda1 * lambda1 * lambda2 * lambda2 * lambda3 * lambda3;
        Matrix4x4 ret = Matrix4x4.zero;
        for (int i = 0; i < 3; i++)
        {
            ret[i, i] = this.stiffness_0 * (l[i] - 1f / l[i]) + this.stiffness_1 * 0.5f * Mathf.Log(i3) / l[i];
        }
        ret[3, 3] = 1f;
        return ret;
    }

    Matrix4x4 get_stress_tensor_mooney_rivlin_peridyno(float lambda1, float lambda2, float lambda3)
    {
        // s: 5000, 5000, 500
        // https://github.com/peridyno/peridyno/blob/7a32a01e33f13d299b77ffd9b6112e2bdff32c46/src/Dynamics/Cuda/Peridynamics/EnergyDensityFunction.h#L478
        float[] l = { lambda1, lambda2, lambda3 };
        float i1 = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
        float i2 = Mathf.Pow(lambda1, 4) + Mathf.Pow(lambda2, 4) + Mathf.Pow(lambda3, 4);
        float i3 = lambda1 * lambda1 * lambda2 * lambda2 * lambda3 * lambda3;
        float j = lambda1 * lambda2 * lambda3;
        float cb3 = Mathf.Pow(j, -2f / 3f);
        Matrix4x4 ret = Matrix4x4.zero;
        for (int i = 0; i < 3; i++)
        {
            float l1 = l[i];
            float l2 = l[(i + 1) % 3];
            float l3 = l[(i + 2) % 3];
            float jn34_derivative = l2 * l3 * (-4f / 3f) * Mathf.Pow(j, -7f / 3f);
            float i1_2_m_i2_derivative = 2 * l1 * (l2 * l2 + l3 * l3);
            // 抵抗膨胀
            ret[i, i] = this.stiffness_0 * (l2 * l3 * (-2f / 3f) * Mathf.Pow(j, -5f / 3f) * i1 + cb3 * 2 * l1)
                // 抵抗膨胀
                + this.stiffness_1 * l2 * l3 * 2 * (j - 1)
                // 抵抗压缩
                + this.stiffness_2 * (jn34_derivative * (i1 * i1 - i2) + Mathf.Pow(j, -4 / 3f) * i1_2_m_i2_derivative);
        }

        ret[3, 3] = 1f;
        return ret;
    }

    Matrix4x4 get_stress_tensor_mooney_rivlin_wiki(float lambda1, float lambda2, float lambda3)
    {
        // 5000, 5000, 2000
        // https://en.wikipedia.org/wiki/Mooney%E2%80%93Rivlin_solid
        float[] l = { lambda1, lambda2, lambda3 };
        Matrix4x4 ret = Matrix4x4.zero;
        float j = lambda1 * lambda2 * lambda3;
        float jn23 = Mathf.Pow(j, -2f / 3f);
        float jn43 = Mathf.Pow(j, -4f / 3f);
        float i1 = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
        float i2 = lambda1 * lambda1 * lambda2 * lambda2 + lambda2 * lambda2 * lambda3 * lambda3 + lambda3 * lambda3 * lambda1 * lambda1;
        for (int k = 0; k < 3; k++)
        {
            float l1 = l[k];
            float l2 = l[(k + 1) % 3];
            float l3 = l[(k + 2) % 3];
            float jn43_p = l2 * l3 * (-4f / 3f) * Mathf.Pow(j, -7f / 3f);
            float i2_p = 2 * l1 * (l2 * l2 + l3 * l3);
            float jn23_p = l2 * l3 * (-2f / 3f) * Mathf.Pow(j, -5f / 3f);
            float i1_p = 2 * l1;
            float j_p = l2 * l3;
            ret[k, k] = this.stiffness_0 * (jn43_p * i2 + jn43 * i2_p) +
                this.stiffness_1 * (jn23_p * i1 + jn23 * i1_p) +
                this.stiffness_2 * j_p * 2 * (j - 1);
        }
        ret[3, 3] = 1f;
        return ret;
    }

    Matrix4x4 get_stress_tensor_fung_peridyno(float lambda1, float lambda2, float lambda3)
    {
        // 5000, 2000, 2000, 2
        float[] l = { lambda1, lambda2, lambda3 };
        float i1 = lambda1 * lambda1 + lambda2 * lambda2 + lambda3 * lambda3;
        float i2 = Mathf.Pow(lambda1, 4) + Mathf.Pow(lambda2, 4) + Mathf.Pow(lambda3, 4);
        float i3 = lambda1 * lambda1 * lambda2 * lambda2 * lambda3 * lambda3;
        float j = lambda1 * lambda2 * lambda3;
        float cb3 = Mathf.Pow(j, -2f / 3f);
        Matrix4x4 ret = Matrix4x4.zero;
        for (int k = 0; k < 3; k++)
        {
            float l1 = l[k];
            float l2 = l[(k + 1) % 3];
            float l3 = l[(k + 2) % 3];
            float cb3_p = (-2f / 3f) * Mathf.Pow(j, -5f / 3f) * l2 * l3;
            float i1_p = 2 * l1;
            // 抵抗膨胀
            ret[k, k] = this.stiffness_0 * (l2 * l3 * (-2f / 3f) * Mathf.Pow(j, -5f / 3f) * i1 + cb3 * 2 * l1)
                // 抵抗膨胀
                + this.stiffness_1 * l2 * l3 * 2 * (j - 1)
                // 抵抗压缩
                + (cb3_p * i1 + cb3 * i1_p)
                    * this.stiffness_3
                    * Mathf.Exp(
                        this.stiffness_3 * (cb3 * i1 - 3) - 1
                    )
                    * this.stiffness_2;
        }

        ret[3, 3] = 1f;
        return ret;
    }

    Matrix4x4 get_first_pk_stress_hyper(Matrix4x4 deformation_gradient)
    {
        Matrix4x4 u_mat = Matrix4x4.zero;
        Matrix4x4 lambda_mat = Matrix4x4.zero;
        Matrix4x4 v_mat = Matrix4x4.zero;
        SVD svd_solver = new SVD();
        svd_solver.svd(deformation_gradient, ref u_mat, ref lambda_mat, ref v_mat);
        if (this.debug)
        {
            // Debug.Log("deformation_gradient: " + deformation_gradient);
            // Debug.Log("u_mat: " + u_mat);
            // Debug.Log("lambda_mat: " + lambda_mat);
            // Debug.Log("v_mat: " + v_mat);
        }
        // https://github.com/peridyno/peridyno/blob/7a32a01e33f13d299b77ffd9b6112e2bdff32c46/src/Dynamics/Cuda/Peridynamics/EnergyDensityFunction.h#L120
        Matrix4x4 stress_tensor;
        switch (this.hyper_model_name)
        {
            case "StVK":
                stress_tensor = this.get_stress_tensor_stvk(lambda_mat[0, 0], lambda_mat[1, 1], lambda_mat[2, 2]);
                break;
            case "NeoHookean":
                stress_tensor = this.get_stress_tensor_neo_hookean(lambda_mat[0, 0], lambda_mat[1, 1], lambda_mat[2, 2]);
                break;
            case "MooneyRivlin1":
                stress_tensor = this.get_stress_tensor_mooney_rivlin_peridyno(lambda_mat[0, 0], lambda_mat[1, 1], lambda_mat[2, 2]);
                break;
            case "MooneyRivlin-Wiki":
                stress_tensor = this.get_stress_tensor_mooney_rivlin_wiki(lambda_mat[0, 0], lambda_mat[1, 1], lambda_mat[2, 2]);
                break;
            case "Fung":
                stress_tensor = this.get_stress_tensor_fung_peridyno(lambda_mat[0, 0], lambda_mat[1, 1], lambda_mat[2, 2]);
                break;
            default:
                throw new Exception("Unknown hyper model: " + this.hyper_model_name);
        }

        // https://github.com/peridyno/peridyno/blob/7a32a01e33f13d299b77ffd9b6112e2bdff32c46/src/Dynamics/Cuda/Peridynamics/Module/CoSemiImplicitHyperelasticitySolver.cu#L661
        var pk1 = u_mat * stress_tensor * v_mat.transpose;
        return pk1;
    }

    void particle_collision()
    {
        for (int i = 0; i < num_nodes; i++)
        {
            // (Particle) collision with floor.
            var dis = this.X[i].y - (-3f);
            if (dis < 0f)
            {
                var hit_normal = new Vector3(0f, 1f, 0f);
                var rel_v = this.V[i];
                var v_ni = Vector3.Dot(rel_v, hit_normal) * hit_normal;
                var v_ti = rel_v - v_ni;
                // 切向速度衰减系数
                var a = Mathf.Max(1 - this.col_restitution * (1 + this.col_restitution)
                    * Vector3.SqrMagnitude(v_ni) / Vector3.SqrMagnitude(v_ti), 0);
                var v_ni_new = -Mathf.Min(1f, this.col_restitution) * v_ni;
                var v_ti_new = a * v_ti;
                this.V[i] = v_ni_new + v_ti_new;
                var x_to_be = this.X[i] - dis * hit_normal;
                this.X[i] = x_to_be;
            }
        }
    }

    void superficial_collision()
    {
        // 已经统计好弹力
        var collision_force = new Vector3[this.num_nodes];
        for (int i = 0; i < this.num_nodes; i++)
        {
            // (Particle) collision with floor.
            var dis = this.X[i].y - (-3f);
            // 一帧上一帧下怎么办
            // < 0 时施加弹力
            // < 0.01 时施加摩擦力
            if (dis < 0.01f)
            {
                var hit_normal = new Vector3(0f, 1f, 0f);
                var rel_v = this.V[i];
                var v_ni = Vector3.Dot(rel_v, hit_normal) * hit_normal;
                var v_ti = rel_v - v_ni;
                // 切向速度衰减系数
                var a = Mathf.Max(1 - this.col_restitution * (1 + this.col_restitution)
                    * Vector3.SqrMagnitude(v_ni) / Vector3.SqrMagnitude(v_ti), 0);
                var v_ni_new = -Mathf.Min(1f, this.col_restitution) * v_ni;
                var v_ti_new = a * v_ti;
                var v_new = v_ni_new + v_ti_new;
                // 压力的弹力
                var elastic_force = (v_ni_new - v_ni) * this.mass / this.dt;
                // collision_force[i] += force_to_this_particle;
                var tangent_static = v_ti.magnitude < 0.05f;
                if (tangent_static)
                {
                    // 静摩擦
                    var outer_force_n = Vector3.Dot(this.outer_force[i], hit_normal) * hit_normal;
                    var outer_force_t = this.outer_force[i] - outer_force_n;
                    if (outer_force_t.magnitude < (elastic_force * this.col_restitution).magnitude)
                    {
                        collision_force[i] += -outer_force_t;
                    }
                    else
                    {
                        collision_force[i] += -(elastic_force * this.col_restitution).magnitude * outer_force_t.normalized;
                    }
                }
                else
                {
                    // 动摩擦
                    collision_force[i] += -(elastic_force * this.col_restitution).magnitude * v_ti.normalized;
                }
                if (dis < 0f)
                {
                    collision_force[i] += elastic_force;
                }
                var x_to_be = this.X[i] - dis * hit_normal;
                this.X[i] = x_to_be;
            }
        }
        for (int i = 0; i < num_nodes; i++)
        {
            this.outer_force[i] += collision_force[i];
        }
    }


    void _Update()
    {
        // 重力或升力
        if (Input.GetKey(KeyCode.Space))
        {
            for (int i = 0; i < num_nodes; i++)
                this.V[i].y += 19.6f * this.dt * (10f / this.calculation_per_frame);
        }
        else
        {
            for (int i = 0; i < num_nodes; i++)
                this.V[i].y -= 9.8f * this.dt * (10f / this.calculation_per_frame);
        }

        // F 要清空
        for (int i = 0; i < this.num_nodes; i++)
        {
            this.outer_force[i] = new Vector3(0, 0, 0);
        }

        if (this.selected_x_id != -1)
        {
            // 鼠标选中了一个点
            Vector3 mouse_pos = Input.mousePosition;
            mouse_pos.z = this.z_when_selected;
            Vector3 world_pos = Camera.main.ScreenToWorldPoint(mouse_pos);
            float elastic_len = (world_pos - this.X[this.selected_x_id]).magnitude; // - this.radius;
            this.outer_force[this.selected_x_id] +=
                (world_pos - this.X[this.selected_x_id]).normalized * elastic_len * this.mass * this.num_nodes * 9.8f;

            LineRenderer line_renderer = GetComponent<LineRenderer>();
            line_renderer.startColor = Color.green;
            line_renderer.SetPositions(new Vector3[] { world_pos, this.X[this.selected_x_id] });
        }

        for (int tet = 0; tet < tet_number; tet++)
        {
            //TODO: Deformation Gradient
            Matrix4x4 deformed_edge_mat = Matrix4x4.zero;
            deformed_edge_mat[3, 3] = 1f;
            for (int i = 0; i < 3; i++)
            {
                deformed_edge_mat.SetColumn(i,
                    this.X[this.Tet[tet * 4 + i + 1]] - this.X[this.Tet[tet * 4 + 0]]);
            }
            var deformation_gradient = deformed_edge_mat * this.inv_dm[tet];

            // this.debug = tet == 12;
            var first_pk_stress = this.use_hyper
                ? this.get_first_pk_stress_hyper(deformation_gradient)
                : this.get_first_pk_stress1(deformation_gradient);
            if (this.debug)
            {
                // var first_pk_stress2 = this.get_first_pk_stress1(deformation_gradient);
                // var first_pk_stress = this.get_first_pk_stress_hyper(deformation_gradient);
                // Debug.Log("first_pk_stress: " + first_pk_stress);
                // Debug.Log("first_pk_stress2: " + first_pk_stress2);
            }

            // Elastic Force
            var f_mat = Mats.dot(-1f / (6f * this.inv_dm[tet].determinant), first_pk_stress * this.inv_dm[tet].transpose);
            var f = new Vector3[4];
            for (int i = 0; i < 3; i++)
            {
                f[i + 1] = f_mat.GetColumn(i);
            }
            f[0] = -f[1] - f[2] - f[3];
            for (int i = 0; i < 4; i++)
            {
                this.outer_force[this.Tet[tet * 4 + i]] += f[i];
            }
        }

        if (this.use_superficial_collision) this.superficial_collision(); else this.particle_collision();

        for (int i = 0; i < num_nodes; i++)
        {
            this.V[i] += this.outer_force[i] * this.dt / this.mass;
            this.damp = Mathf.Pow(0.999f, this.dt / 0.00166f);
            this.V[i] *= this.damp;
        }

        // Laplacian smoothing.
        for (int i = 0; i < num_nodes; i++)
        {
            this.V_sum[i] = new Vector3(0, 0, 0);
        }
        for (int i = 0; i < tet_number; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                for (int k = 0; k < 4; k++)
                {
                    this.V_sum[this.Tet[i * 4 + j]] += this.V[this.Tet[i * 4 + k]];
                }
            }
        }
        for (int i = 0; i < num_nodes; i++)
        {
            this.V[i] = this.V_sum[i] / this.ref_count[i];
        }

        for (int i = 0; i < num_nodes; i++)
        {
            this.X[i] += this.V[i] * this.dt;
        }
    }

    void handle_mouse_movement()
    {
        // 鼠标
        if (Input.GetMouseButtonDown(0))
        {
            this.mouse_pressed = true;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
            float closest_z = float.MaxValue;
            this.selected_x_id = -1;
            for (int i = 0; i < this.X.Length; i++)
            {
                // float proj_dis = Vector3.Dot(ray.direction.normalized, this.X[i] - ray.origin);
                // float dis = Vector3.Distance(ray.origin + proj_dis * ray.direction.normalized, this.X[i]);
                float dis = Vector3.Cross(ray.direction, this.X[i] - ray.origin).magnitude;
                if (dis < this.average_edge_length)
                {
                    float z = Camera.main.WorldToScreenPoint(this.X[i]).z;
                    if (closest_z > z)
                    {
                        closest_z = z;
                        this.selected_x_id = i;
                        this.z_when_selected = z;
                        this.GetComponent<LineRenderer>().enabled = true;
                    }
                }
            }
        }
        if (Input.GetMouseButtonUp(0))
        {
            this.GetComponent<LineRenderer>().enabled = false;
            this.mouse_pressed = false;
            this.selected_x_id = -1;
        }
    }

    // Update is called once per frame
    void Update()
    {
        this.hyper_model_name = this.hyper_model_name_map[this.hyper_model];
        if (this.hyper_model != 0)
        {
            this.use_hyper = true;
        }
        this.handle_mouse_movement();

        for (int l = 0; l < this.calculation_per_frame; l++)
        {
            // 每帧时间其实是 10 dt
            _Update();
        }

        // 上面只更新 X 不更新 mesh
        // Dump the vertex array for rendering.
        Vector3[] vertices = new Vector3[tet_number * 12];
        int vertex_number = 0;
        for (int tet = 0; tet < tet_number; tet++)
        {
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 0]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 1]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 2]];
            vertices[vertex_number++] = X[Tet[tet * 4 + 3]];
        }
        Mesh mesh = GetComponent<MeshFilter>().mesh;
        mesh.vertices = vertices;
        mesh.RecalculateNormals();
    }
}
