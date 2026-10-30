// Synchronous, headless adapter to real Stonefish physics (no ROS required).
// stdin: RESET x y z yaw | STEP <physics ticks> <six normalized PWM> | QUIT
// stdout: HOFA_STATE <time> <x y z> <u v w> <qx qy qz qw> <p q r>
// Other Stonefish console lines are captured in the Python backend log.
#include <Stonefish/core/ConsoleSimulationApp.h>
#include <Stonefish/core/ScenarioParser.h>
#include <Stonefish/core/SimulationManager.h>
#include <Stonefish/core/Robot.h>
#include <Stonefish/actuators/Thruster.h>
#include <Stonefish/sensors/scalar/Odometry.h>
#include <Stonefish/sensors/Sample.h>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

class Manager : public sf::SimulationManager {
public:
    Manager(double rate, const std::string& scene)
        : sf::SimulationManager(rate, sf::Solver::SI, sf::CollisionFilter::EXCLUSIVE),
          scene_(scene) {}
    void BuildScenario() override {
        sf::ScenarioParser parser(this);
        if (!parser.Parse(scene_)) throw std::runtime_error("Stonefish scenario parsing failed");
    }
    void SimulationStepCompleted(sf::Scalar) override {}
private:
    std::string scene_;
};

class App : public sf::ConsoleSimulationApp {
public:
    App(const std::string& data, Manager* manager)
        : sf::ConsoleSimulationApp("HOFA residual RL", data, manager) {
        // No background physics thread: only a STEP command advances physics.
        autostep_ = false;
        setMaxPhysicsThreads(1);
    }
    void initialize() { Init(); }
};

static void state(Manager& manager) {
    auto* odom = dynamic_cast<sf::Odometry*>(manager.getSensor("bricsbot/odometry"));
    if (!odom) throw std::runtime_error("missing bricsbot/odometry sensor");
    const sf::Sample sample = odom->getLastSample();
    std::ostringstream line;
    line << std::setprecision(17) << "HOFA_STATE " << manager.getSimulationTime();
    for (int i = 0; i < 13; ++i) line << ' ' << sample.getValue(i);
    std::cout << line.str() << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "usage: hofa_stonefish_step DATA_DIR SCENARIO PHYSICS_HZ\n";
        return 2;
    }
    try {
        const double rate = std::stod(argv[3]);
        if (!std::isfinite(rate) || rate < 20 || rate > 2000)
            throw std::runtime_error("invalid physics rate");
        Manager manager(rate, argv[2]);
        App app(std::string(argv[1]) + "/", &manager);
        app.initialize();
        manager.setCallSimulationStepCompleted(false);
        std::cout << "HOFA_READY 1" << std::endl;
        std::string input;
        bool started = false;
        std::array<sf::Thruster*, 6> thrusters{};
        while (std::getline(std::cin, input)) {
            std::istringstream command(input);
            std::string name;
            command >> name;
            if (name == "QUIT") break;
            if (name == "RESET") {
                double x, y, z, yaw;
                if (!(command >> x >> y >> z >> yaw) ||
                    !std::isfinite(x+y+z+yaw)) throw std::runtime_error("invalid RESET");
                // Rebuild the world to clear velocities, actuator dynamics,
                // sensor histories, contacts and the physics clock together.
                if (started) manager.StopSimulation();
                manager.RestartScenario();
                manager.setCallSimulationStepCompleted(false);
                auto* robot = manager.getRobot("bricsbot");
                if (!robot) throw std::runtime_error("missing bricsbot robot");
                sf::Quaternion orientation;
                orientation.setEulerZYX(yaw, 0, 0);
                robot->Respawn(&manager, sf::Transform(orientation, sf::Vector3(x,y,z)));
                for (int i=0; i<6; ++i) {
                    thrusters[i] = dynamic_cast<sf::Thruster*>(robot->getActuator("bricsbot/T" + std::to_string(i+1)));
                    if (!thrusters[i]) throw std::runtime_error("missing thruster T" + std::to_string(i+1));
                    thrusters[i]->setSetpoint(0);
                }
                if (!manager.StartSimulation()) throw std::runtime_error("initial condition solve failed");
                started = true;
                // One unforced tick supplies the first real odometry sample.
                manager.StepSimulation(1.0/rate);
                state(manager);
            } else if (name == "STEP") {
                int ticks;
                if (!started || !(command >> ticks) || ticks < 1 || ticks > 2000)
                    throw std::runtime_error("invalid STEP tick count");
                std::array<double,6> pwm;
                for (int i=0; i<6; ++i) {
                    if (!(command >> pwm[i]) || !std::isfinite(pwm[i]) || std::abs(pwm[i])>1)
                        throw std::runtime_error("invalid PWM");
                }
                for (int i=0; i<6; ++i) thrusters[i]->setSetpoint(pwm[i]);
                // One fixed physics tick at a time; learning/inference wall
                // time cannot advance this world.
                for (int i=0; i<ticks; ++i) manager.StepSimulation(1.0/rate);
                state(manager);
            } else {
                throw std::runtime_error("unknown command");
            }
        }
        if (started) manager.StopSimulation();
        manager.DestroyScenario();
        return 0;
    } catch (const std::exception& error) {
        std::cout << "HOFA_ERROR " << error.what() << std::endl;
        return 1;
    }
}
