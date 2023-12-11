using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Mats
{
    public static Matrix4x4 zero4()
    {
        return Matrix4x4.zero;
    }
    public static Matrix4x4 cross_mat(Vector3 a)
    {
        Matrix4x4 A = Mats.zero4();
        A[0, 0] = 0;
        A[0, 1] = -a[2];
        A[0, 2] = a[1];
        A[1, 0] = a[2];
        A[1, 1] = 0;
        A[1, 2] = -a[0];
        A[2, 0] = -a[1];
        A[2, 1] = a[0];
        A[2, 2] = 0;
        A[3, 3] = 1;
        return A;
    }

    public static Matrix4x4 dot(Matrix4x4 mat, float x)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                mat[i, j] *= x;
            }
        }
        return mat;
    }

    public static Matrix4x4 dot(float x, Matrix4x4 mat)
    {
        return dot(mat, x);
    }

    public static float trace(Matrix4x4 mat)
    {
        float res = 0;
        for (int i = 0; i < 4; i++)
        {
            res += mat[i, i];
        }
        return res;
    }

    public static Matrix4x4 add(Matrix4x4 m1, Matrix4x4 m2)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
            {
                m1[i, j] += m2[i, j];
            }
        }
        return m1;
    }

    public static Matrix4x4 minus(Matrix4x4 m1, Matrix4x4 m2)
    {
        return add(m1, dot(m2, -1f));
    }

    public static Matrix4x4 vec_vec_mat(Vector3 a, Vector3 b)
    {
        var res = Matrix4x4.zero;
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                res[i, j] = a[i] * b[j];
            }
        }
        return res;
    }

    public static Quaternion r_mat_to_q(Matrix4x4 m)
    {
        return Quaternion.LookRotation(m.GetColumn(2), m.GetColumn(1));
    }

    public static Matrix4x4 q_to_r_mat(Quaternion q)
    {
        return Matrix4x4.TRS(Vector3.zero, q, Vector3.one);
    }
}
