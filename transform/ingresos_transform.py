from pathlib import Path
import sys
import pandas as pd
from datetime import datetime
from typing import Optional, List, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utilidades.etl_date_utils import DateUtilsETL
from configs.ingresos_config import IngresosConfig
from transform.procesadores._calculator_ingresos import IngresosCalculator, ContinuoIngresosCalculator, DesviosIngresosCalculator

class TransformadorIngresos:
    def __init__(self):
        self.date_utils = DateUtilsETL()
        self.config = IngresosConfig()

        self.transform_types = ['latest', 'single', 'multiple']

        self.market_calculator_map = {
            'continuo': ContinuoIngresosCalculator,
            'desvios': DesviosIngresosCalculator
        }


    def get_calculator_for_market(self, market_key: str):
        calculator_class = self.market_calculator_map.get(market_key, IngresosCalculator)
        return calculator_class()

    def calculate_ingresos_for_all_markets(self, fecha_inicio: Optional[str] = None, fecha_fin: Optional[str] = None,
                                           mercados_lst: Optional[List[str]] = None, plot: bool = False) -> dict:
        
        #get valid markets ie (keys in mercado_name_id_map)
        valid_markets = list(self.config.mercado_name_id_map.keys())

        if mercados_lst is None:
            mercados_lst = valid_markets
        else:
            invalid_markets = [m for m in mercados_lst if m not in valid_markets]
            if invalid_markets:
                print(f"Warning: Invalid markets {invalid_markets} skipped.")
            mercados_lst = [m for m in mercados_lst if m in valid_markets]

        if fecha_inicio is None and fecha_fin is None:
            raise ValueError("Fecha inicio and fecha fin are required")
        elif fecha_inicio is not None and (fecha_fin is None or fecha_inicio == fecha_fin):
            transform_type = 'single'
        elif fecha_inicio is not None and fecha_fin is not None and fecha_inicio != fecha_fin:
            transform_type = 'multiple'
        else:
            raise ValueError("Invalid date parameters.")

        status_details = {
            "markets_processed": [],
            "markets_failed": [],
            "mode": transform_type,
            "date_range": f"{fecha_inicio} to {fecha_fin}" if fecha_fin else fecha_inicio
        }
        overall_success = True
        results = {}

        print(f"\n===== STARTING INGRESOS CALCULATION (Mode: {transform_type.upper()}) =====")
        for market_key in mercados_lst:
            print(f"\n-- Market: {market_key} --")
            try:
                calculator = self.get_calculator_for_market(market_key)
                if transform_type == 'single':
                    market_result = calculator.calculate_single(market_key, fecha_inicio, plot)
                elif transform_type == 'multiple':
                    market_result = calculator.calculate_multiple(market_key, fecha_inicio, fecha_fin, plot)

                if isinstance(market_result, pd.DataFrame) and not market_result.empty:
                    status_details["markets_processed"].append(market_key)
                    results[market_key] = market_result
                else:
                    status_details["markets_failed"].append({"market": market_key, "error": "No data produced"})
                    overall_success = False
            except Exception as e:
                status_details["markets_failed"].append({"market": market_key, "error": str(e)})
                overall_success = False
                results[market_key] = None
                print(f"❌ Failed for {market_key}: {e}")

        print(f"\n===== INGRESOS CALCULATION FINISHED (Mode: {transform_type.upper()}) =====")
        for market_key, result in results.items():
            if result is not None:
                print(f"✅ Success for {market_key}")
                print(result.head())
                print(result.tail())
                print(f"\n")
            else:
                print(f"❌ Failed for {market_key}")

        return {
            "data": results,
            "status": {
                "success": overall_success,
                "details": status_details
            }
        }


