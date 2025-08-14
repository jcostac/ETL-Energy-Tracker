import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from tecnologias_generacion.p48_tecnologias_generacion import P48TecnologiasGeneracion

def test_update_tecnologias_generacion():
    p48_tecnologias = P48TecnologiasGeneracion()

    unique_p48_tecnologias = p48_tecnologias.get_unique_p48_conceptos()

    result = p48_tecnologias.update_tecnologias_generacion(unique_p48_tecnologias)
    assert result, "New tecnologias were not inserted into the database."

if __name__ == "__main__":
    test_update_tecnologias_generacion()

