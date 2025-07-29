from pathlib import Path
from datetime import datetime, timedelta

class IngresosConfig:
    """Configuration for the ingresos module"""

    def __init__(self):

        # Set the cutoff date for SRS market reduction
        self.dia_inicio_SRS = datetime(2024, 11, 20) 

        # Set the cutoff date for intra market reduction
        self.intra_reduction_date = datetime(2024, 6, 13)

        self.mercado_name_id_map = {
            'diario': [1],
            'intra': [2, 3, 4, 5, 6, 7, 8],
            'continuo': [21],
            'secundaria': [14,15],
            'terciaria': [18,19,26,27],
            'rr': [16,17],
            'curtailment': [13,23],
            'p48': [20],
            'indisponibilidades': [22],
            'restricciones': [9,10,11,12, 24, 25],
        }

    def get_precios_from_id_mercado(self, id_mercado: str, date: datetime) -> int:
        """
        Returns the id of the corresponding prices sheet for a given id_mercado.
        """

        # ==== DIARIO + INTRAS (PHF Y MIC) + RESTRICCIONES MD/TR (subir/bajar) + TERCIARIA DIRECTA (subir/bajar) + RT2 (subir/bajar)====
        if (1 <= id_mercado <= 12) or id_mercado in [21, 24, 25, 26, 27]:
            return id_mercado

        # ==== CURTAILMENT, P48, CURTAILMENT DEMANDA, INDISPONIBILIDADES====
        elif id_mercado in [13, 20, 22, 23]:
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

        else:
            raise ValueError(f"Invalid market id: {id_mercado}, market not found in ingresos config")

    

            

