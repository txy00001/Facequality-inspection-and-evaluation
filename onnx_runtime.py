


# ##检查onnx
# onnx_model = onnx.load("onnx/esnet.onnx")
# onnx.checker.check_model(onnx_model)
# print(onnx.helper.printable_graph(onnx_model.graph))
# ###精简onnx
# model = onnx.load("onnx/esnet.onnx")  # load your predefined ONNX model
# model_simp, check = simplify(model)  # convert model
#
# assert check, "Simplified ONNX model could not be validated"

###推理onnx
import onnxruntime as ort
import cv2
import numpy as np
import time
import onnxruntime

model_path =r"E:\face_project\Face classification\onnx\esnet_new_sim.onnx"
session_option = onnxruntime.SessionOptions()
# session_option.optimized_model_filepath = f"{model_file_name}_cudaopt.onnx"
session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
# session_option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
session_option.log_severity_level = 3
# onnxruntime.capi._pybind_state.set_openvino_device('CPU_FP32')
session = onnxruntime.InferenceSession(
      model_path,
      session_option,
      providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
print("==>> input_name: ", input_name)  # input
output_names = [o.name for o in session.get_outputs()]
print("==>> output_names: ", output_names)
input_shape = session.get_inputs()[0].shape
print("==>> input_shape: ", input_shape)

img = cv2.imread("test.jpg")

resized_imgs = cv2.resize(img, (128, 128), interpolation=cv2.INTER_CUBIC)
# print("==>> resized_img.shape: ", resized_imgs.shape)
resized_img = np.transpose(resized_imgs, (2, 0, 1))
# print("==>> trans_img.shape: ", resized_img.shape)
resized_img = np.expand_dims(resized_img, axis=0)
# print("==>> expand_img.shape: ", resized_img.shape)
dummy_input = np.array(resized_img).astype(np.float32) / 255.0
dummy_input = (dummy_input - 0.5433) / 0.2005

output = session.run(
      output_names,
      {input_name: dummy_input}
)
s1 = time.time()
for i in range(1):
      pred_onnx = session.run(output_names,
                              {input_name: dummy_input})

# resized_imgs= resized_img.astype(np.uint8)
cv2.putText(resized_imgs.astype(np.float32), "face", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
           2)  # 显示模型名
t1 = time.time()
cv2.imshow("Save", resized_imgs)
cv2.waitKey(0)
print("用onnx完成1次推理消耗的时间:%s" % (t1 - s1))