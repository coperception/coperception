#include <pybind11/eigen/matrix.h>
#include <pybind11/stl.h>

struct UnknownParam {
  int x;
  int y;
  int z;
};

long long sub_7FF446E16480(std::vector<UnknownParam> &a1,
                           const Eigen::Matrix<float, Eigen::Dynamic, 1> &a2,
                           const Eigen::Matrix<float, Eigen::Dynamic, 1> &a3,
                           const Eigen::Matrix<float, Eigen::Dynamic, 1> &a4,
                           float a5) {
  auto v66 =
      UnknownParam{(int)std::floor(a2[0] / a5), (int)std::floor(a2[1] / a5),
                   (int)std::floor(a2[2] / 0.4)};
  double v29 = floor(a3[1] / a5);
  double v61 = floor(a3[2] / 0.4);
  double v32 = floor(a3[0] / a5);
  double v63 = 0.4 / abs(a3[2] - a2[2]);
  double v65 = a5 / abs(a3[1] - a2[1]);
  double v64 = a5 / abs(a3[0] - a2[0]);

  int v35 = a3[0] < a2[0] ? -1 : 1;
  int v36 = a3[1] < a2[1] ? -1 : 1;
  int v38 = a3[2] < a2[2] ? -1 : 1;
  double v39 = ((floor(a2[0] / a5) + v35) * a5 - a2[0]) / (a3[0] - a2[0]);
  double v40 = ((floor(a2[1] / a5) + v36) * a5 - a2[1]) / (a3[1] - a2[1]);
  double v41 = ((floor(a2[2] / 0.4) + v38) * 0.4 - a2[2]) / (a3[2] - a2[2]);

  double v23;
  a1.push_back(v66);

  bool new_item = false;
  if ((floor(a2[0] / a5) != floor(a3[0] / a5) && a3[0] < a2[0])) {
    v66.x -= 1;
    new_item = true;
  }
  if ((floor(a2[1] / a5) != floor(a3[1] / a5) && a3[1] < a2[1])) {
    v66.y -= 1;
    new_item = true;
  }
  if ((floor(a2[2] / 0.4) != floor(a3[2] / 0.4) && a3[2] < a2[2])) {
    v66.z -= 1;
    new_item = true;
  }
  if (new_item) {
    a1.push_back(v66);
  }
  v23 = a1.back().x;
  while (!(v32 == v23 && v29 == v66.y && v61 == v66.z)) {
    if (std::min(v40, v41) > v39) {
      v23 += v35;
      v66.x = v23;
      if (a4[0] <= v23 || v23 < 0)
        return 1LL;
      v39 += v64;
    } else if (v40 > v39 && v39 >= v41 || v41 <= v40) {
      v66.z += v38;
      if (v66.z < 0 || a4[2] <= v66.z)
        return 1LL;
      v41 += v63;
    } else {
      v66.y += v36;
      if (v66.y < 0 || a4[1] <= v66.y)
        return 1LL;
      v40 += v65;
    }
    a1.push_back(v66);
    v23 = v66.x;
  }
  return 0;
}
Eigen::Matrix<float, Eigen::Dynamic, 1>
compute_logodds_dp(Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &a2,
                   Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> &a3,
                   Eigen::Matrix<float, Eigen::Dynamic, 1> &a4,
                   std::vector<int> &a5, double a6, double a7, double a8) {
  int v14 = (int)((a4[3] - a4[0]) / a6);
  int v15 = (int)((a4[4] - a4[1]) / a6);
  double v16 = std::ceil((a4[5] - a4[2]) / 0.4);
  Eigen::Matrix<float, Eigen::Dynamic, 1> ret =
      Eigen::MatrixXf::Zero(v14 * v15 * v16, 1);
  auto v55 = Eigen::Vector3f(v14, v15, v16);
  auto v56 = a3.row(0).transpose() - a4.block<3, 1>(0, 0);

  for (auto index : a5) {

    auto v57 = a2.row(index).transpose() - a4.block<4, 1>(0, 0);
    std::vector<UnknownParam> params;
    int result = sub_7FF446E16480(params, v56, v57, v55, a6);
    if (!params.empty()) {
      for (int i = 0; i < params.size(); i++) {
        auto &param = params[i];
        int v36 = param.x + v14 * param.y + v14 * v15 * param.z;
        if (ret[v36] <= 0) {
          if (i == params.size() - 1 && !result) {
            ret[v36] = a7;
          } else {
            ret[v36] = a8;
          }
        }
      }
    }
  }
  return ret;
}
PYBIND11_MODULE(mapping, m) {
  m.def("compute_logodds_dp", compute_logodds_dp,
        pybind11::arg("original_points"), pybind11::arg("sensor_origins"),
        pybind11::arg("pc_range"), pybind11::arg("indices"),
        pybind11::arg("voxel_size"),
        pybind11::arg("lo_occupied") = 0.8472978603872034,
        pybind11::arg("lo_free") = -0.4054651081081643);
  m.doc() = "mapping rewrite";
}
