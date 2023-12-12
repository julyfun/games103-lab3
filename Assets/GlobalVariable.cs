
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

// 所有物体共用的参数 
public class GlobalVariable : MonoBehaviour
{
    [Header("全局变量")]
    [Range(0.001666f, 0.005f)]
    public float dt = 0.001666f;
    [Range(0f, 1f)]
    public float floor_restitution = 0.5f;
    public bool enable_stiction = true;
    [Range(1, 10)]
    public int calculation_per_frame = 5;
    [Range(5f, 98)]
    public float gravity = 29.7f;
}
