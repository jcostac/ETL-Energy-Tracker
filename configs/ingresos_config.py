from pathlib import Path
from datetime import datetime, timedelta

class IngresosConfig:
    """Configuration for the ingresos module"""

    def __init__(self):
        self.dia_inicio_SRS = datetime(2024, 11, 20) 

        self.mercados_precios_map = None

    def get_precios_from_id_mercado(self, id_mercado: str, date: datetime) -> int:
        """
        Returns the id of the corresponding prices sheet for a given id_mercado.
        """

        # ==== DIARIO + INTRAS + RESTRICCIONES MD/TR (subir/bajar) + TERCIARIA DIRECTA (subir/bajar) + RT2 (subir/bajar)====
        if (1 <= id_mercado <= 12) or id_mercado in {24, 25, 26, 27}:
            return id_mercado

        # ==== CURTAILMENT, P48, CURTAILMENT DEMANDA ====
        elif id_mercado in [13, 20, 23]:
            return 1 


        # === RR ====
        elif id_mercado in [16, 17]:
            return 16 #precio unificado RR

        # ==== SECUNDARIA ====
        elif id_mercado in [14, 15]:
            if date < self.dia_inicio_SRS: #precio unificado secundaria
                return 15 
            else:
                return id_mercado #precio dual secundaria (14, 15)

        # ==== TERCIARIA ====
        elif id_mercado in [18, 19]: 
            if date < self.dia_inicio_SRS:
                return id_mercado #precio dual terciaria (18, 19)
            else: 
                return 28 #precio unificado terciaria 

        # === TERCIARIA DIRECTA ====
        elif id_mercado in [26, 27]:
            return id_mercado

    

            

