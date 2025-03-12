from python_testing.circuit_models.demo_cnn import Demo
from python_testing.circuit_models.doom_model_2 import Doom as Doom2
from python_testing.circuit_models.doom_model import Doom as Doom1
from python_testing.circuit_models.eth_fraud import Eth
from python_testing.circuit_models.simple_circuit import SimpleCircuit
from python_testing.circuit_models.testing_circuit_doom import Doom as DoomOriginal




if __name__ == "__main__":
    Demo().run_circuit()
    Doom2().run_circuit()
    Doom1().run_circuit()
    SimpleCircuit().base_testing()
    DoomOriginal().run_circuit()


