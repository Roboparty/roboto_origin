#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "robot_interface.hpp"

namespace py = pybind11;

PYBIND11_MODULE(robot_py, m) {
    py::class_<RobotInterface>(m, "RobotInterface")
        .def(py::init<const std::string&>(), py::arg("config_file"))
        .def("apply_action", &RobotInterface::apply_action, py::arg("action"))
        .def("init_motors", &RobotInterface::init_motors)
        .def("deinit_motors", &RobotInterface::deinit_motors)
        .def("reset_motors", &RobotInterface::reset_motors, py::arg("joint_default_angle"))
        .def("set_zeros", &RobotInterface::set_zeros)
        .def("clear_errors", &RobotInterface::clear_errors)
        .def("read_motors", &RobotInterface::read_motors)
        .def("get_joint_q", &RobotInterface::get_joint_q)
        .def("get_joint_vel", &RobotInterface::get_joint_vel)
        .def("get_joint_tau", &RobotInterface::get_joint_tau)
        .def("get_quat", &RobotInterface::get_quat)
        .def("get_ang_vel", &RobotInterface::get_ang_vel)
        .def_property_readonly("is_init", [](const RobotInterface &r) {
            return r.is_init_.load();
        });
}
