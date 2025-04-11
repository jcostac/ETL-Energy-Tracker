from deprecated import deprecated
from read_ops.read_data import ReadOps

class Volumenesi90_Reader(ReadOps):
    """
    Clase para leer datos de volumen de i90 tanto de la base de datos como de los ficheros parquet.
    """
    def __init__(self):
        super().__init__()

@deprecated(action="default", reason="Class used in old ETL pipeline and DB structure, now deprecated")
class Volumenesi90_DB_Reader(Volumenesi90_Reader):
    """
    Clase para leer datos de volumen de i90 de la base de datos.
    """
    def __init__(self):
        super().__init__()
        
        
            