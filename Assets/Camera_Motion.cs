using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Camera_Motion : MonoBehaviour
{
    bool pressed = false;

    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetMouseButtonDown(0))
        {
            pressed = true;
            Ray ray = Camera.main.ScreenPointToRay(Input.mousePosition);
        }
        if (Input.GetMouseButtonUp(0))
            pressed = false;

        if (pressed && FVM.selecting_cnt == 0)
        {
            {

                float v = 2.0f * Input.GetAxis("Mouse Y");
                float h = 2.0f * Input.GetAxis("Mouse X");
                Camera.main.transform.RotateAround(new Vector3(0, 0, 0), Vector3.up, h);
                Vector3 cam_forward = Camera.main.transform.forward;
                Vector3 cam_up = Camera.main.transform.up;
                // 垂直移动鼠标时，需要绕一个平行于地面和相机平面的轴旋转
                Vector3 parallel_to_ground_and_camera = Vector3.Cross(cam_forward, cam_up).normalized;
                Camera.main.transform.RotateAround(new Vector3(0, 0, 0), parallel_to_ground_and_camera, v);
                //     float h;

                //     h = 2.0f * Input.GetAxis("Mouse Y");
                //     transform.Rotate(h, 0, 0);

                //     h = 2.0f * Input.GetAxis("Mouse X");
                //     Camera.main.transform.RotateAround(new Vector3(0, 0, 0), Vector3.up, h);
            }
        }
        if (Input.GetKey(KeyCode.W))
        {
            Camera.main.transform.Translate(Vector3.forward);
        }

        if (Input.GetKey(KeyCode.S))
        {
            Camera.main.transform.Translate(Vector3.back);
        }
    }
}
