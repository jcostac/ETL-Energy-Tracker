

def _process_datetime_continuo(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process datetime for continuo market.
        Uses 'Contrato' column to extract delivery period and applies DST adjustments.
        
        Args:
            df (pd.DataFrame): Input DataFrame with 'Contrato' column
            
        Returns:
            pd.DataFrame: DataFrame with delivery_period column in UTC
        """
        print("\n🕐 PROCESSING DATETIME (CONTINUO)")
        print("-"*30)
        
        if 'Fecha' in df.columns:
            # Process file date (fecha_fichero)
            df['fecha'] = pd.to_datetime(df['Fecha'], format="mixed") #some dates have hours ie 00:00:00
            df['fecha_fichero'] = df['fecha'].dt.strftime('%Y-%m-%d')
        
        if 'Contrato' in df.columns:
            # Extract delivery date from contract string (first 8 characters: YYYYMMDD)
            df['delivery_date_str'] = df['Contrato'].str.strip().str[:8]
            df['delivery_date'] = pd.to_datetime(df['delivery_date_str'], format="%Y%m%d")
            
            # Get transition dates for DST handling
            year_min = df['delivery_date'].dt.year.min()
            year_max = df['delivery_date'].dt.year.max()
            start_range = datetime(year_min - 1, 1, 1)
            end_range = datetime(year_max + 1, 12, 31)
            transition_dates = TimeUtils.get_transition_dates(start_range, end_range)
            
            # Apply hour adjustment logic from ajuste_horario function
def _adjust_continuo_hour(self, row, transition_dates: Dict) -> int:
        """
        Adjust hour for continuo market based on DST transitions.
        Implements the ajuste_horario logic from carga_omie.py.
        
        Args:
            row: DataFrame row with delivery date and contract info
            transition_dates: Dictionary of DST transition dates
            
        Returns:
            int: Adjusted hour
        """
        date_ref = row['delivery_date'].date()
        is_special_date = date_ref in transition_dates
        tipo_cambio_hora = transition_dates.get(date_ref, 0)
        
        # Extract hour from contract string (positions 9-11)
        contract_hour = int(row['Contrato'].strip()[9:11])
        
        if is_special_date:
            if tipo_cambio_hora == 2:  # 23-hour day (spring forward)
                if (contract_hour + 1) < 3:
                    return contract_hour + 1
                else:
                    return contract_hour
            
            elif tipo_cambio_hora == 1:  # 25-hour day (fall back)
                if row['Contrato'].strip()[-1].isdigit():
                    if (contract_hour + 1) < 3:
                        return contract_hour + 1
                    else:
                        return contract_hour + 2
                elif row['Contrato'].strip()[-1] == 'A':
                    return contract_hour + 1
                elif row['Contrato'].strip()[-1] == 'B':
                    return contract_hour + 2
        
        # Normal days
        return contract_hour + 1
