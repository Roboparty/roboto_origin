#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "imu_driver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(imu_py, m) {
    m.doc() = "IMU Driver Python SDK"; 

    py::class_<IMUDriver, std::shared_ptr<IMUDriver>>(m, "IMUDriver")
        .def(py::init<>())
        .def_static("create_imu", &IMUDriver::create_imu, 
            py::arg("imu_id"),
            py::arg("interface_type"),
            py::arg("interface"),
            py::arg("imu_type"),
            py::arg("baudrate") = 0)
        .def("get_imu_id", &IMUDriver::get_imu_id)
        .def("get_ang_vel", &IMUDriver::get_ang_vel)
        .def("get_quat", &IMUDriver::get_quat)
        .def("get_lin_acc", &IMUDriver::get_lin_acc)
        .def("get_temperature", &IMUDriver::get_temperature);
}
