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
            'p48': [20],
            'indisponibilidades': [22],
            'restricciones_md': [9,10],
            'restricciones_tr': [11,12],
            'desvios': [29,30]
        }

        self.mercado_sentido_map = {
            'subir': {
                'secundaria': [14],
                'terciaria': [18],
                "terciaria_directa": [26],
                'rr': [16]
            },
            'bajar': {
                'secundaria': [15],
                'terciaria': [19],
                "terciaria_directa": [27],
                'rr': [17]
            }
        }

    def get_market_ids(self, market_key, date):
        """
        Returns the appropriate market IDs for a given market key and date.
        For intra markets, filters based on the intra reduction date.
        """
        if market_key == 'intra':
            if date >= self.intra_reduction_date:
                # After June 13, 2024: only use Intra 1, 2, and 3
                return [2, 3, 4]
            else:
                # Before June 13, 2024: use all 7 intra markets
                return [2, 3, 4, 5, 6, 7, 8]
        else:
            # For all other markets, return the standard mapping
            return self.mercado_name_id_map.get(market_key, [])

    def get_precios_from_id_mercado(self, id_mercado: str, date: datetime) -> int:
        """
        Returns the market id of the corresponding prices dataset for a given id_mercado.
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

        # === DESVIOS ===
        elif id_mercado in [29, 30]:
            return None #needs to be calculated

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

    

            

