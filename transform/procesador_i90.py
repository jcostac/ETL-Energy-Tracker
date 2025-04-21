class SecundariaTransform:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass


    def get_volumenes(self, day: datetime, bbdd_engine, UP_ids: Optional[List[int]] = None, 
                        sentidos: Optional[List[str]] = None) -> List[Tuple[pd.DataFrame, str]]:
            """
            Get tertiary regulation volume data for a specific day.
            
            Args:
                day (datetime): Day to download data for
                bbdd_engine: Database engine connection
                UP_ids (Optional[List[int]]): List of programming unit IDs to filter
                sentidos (Optional[List[str]]): List of directions to filter ['Subir', 'Bajar', 'Directa']
                
            Returns:
                List[Tuple[pd.DataFrame, str]]: List of tuples with (DataFrame, direction)
            """
            # Set default sentidos if None
            if sentidos is None:
                sentidos = ['Subir', 'Bajar', 'Directa']
            
            # Get Programming Units
            unidades, dict_unidades = self.get_programming_units(bbdd_engine, UP_ids)
            
            # Download I90 file
            file_name, file_name_2 = self.download_i90_file(day)
            
            # Get transition dates
            filtered_transition_dates = self.get_transition_dates(day, day + timedelta(days=1))
            day_datef = day.date()
            is_special_date = day_datef in filtered_transition_dates
            tipo_cambio_hora = filtered_transition_dates.get(day_datef, 0)
            
            # Get error data
            df_errores = self.get_error_data(bbdd_engine)
            df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
            pestañas_con_error = df_errores_dia['tipo_error'].values
            
            results = []
            
            # Check if the sheet has errors
            if self.sheet_id not in pestañas_con_error:
                sheet = str(self.sheet_id).zfill(2)
                
                # Load sheet data
                df = pd.read_excel(f"./I90DIA_{file_name_2}.xls", sheet_name=f'I90DIA{sheet}', skiprows=2)
                
                # Filter by programming units
                df = df[df['Unidad de Programación'].isin(unidades)]
                
                # Process for each direction
                for sentido in sentidos:
                    df_sentido = df.copy()
                    
                    # Apply direction filter
                    if sentido == 'Subir':
                        df_sentido = df_sentido[df_sentido['Sentido'] == 'Subir']
                        df_sentido = df_sentido[df_sentido['Redespacho'] != 'TERDIR']
                    elif sentido == 'Bajar':
                        df_sentido = df_sentido[df_sentido['Sentido'] == 'Bajar']
                        df_sentido = df_sentido[df_sentido['Redespacho'] != 'TERDIR']
                    elif sentido == 'Directa':
                        df_sentido = df_sentido[df_sentido['Redespacho'] == 'TERDIR']
                    
                    # Filter columns to keep only programming unit and value columns
                    if not df_sentido.empty:
                        total_col = df_sentido.columns.get_loc("Total")
                        cols_to_drop = list(range(0, total_col+1))
                        up_col = df_sentido.columns.get_loc("Unidad de Programación")
                        cols_to_drop.remove(up_col)
                        df_sentido = df_sentido.drop(df_sentido.columns[cols_to_drop], axis=1)
                        
                        # Melt the dataframe to convert from wide to long format
                        hora_colname = df_sentido.columns[1]
                        df_sentido = df_sentido.melt(id_vars=["Unidad de Programación"], var_name="hora", value_name="volumen")
                        
                        # Apply time adjustments
                        if hora_colname == 1:
                            df_sentido['hora'] = df_sentido.apply(self.ajuste_quinceminutal, axis=1, 
                                                                is_special_date=is_special_date, 
                                                                tipo_cambio_hora=tipo_cambio_hora)
                        else:
                            df_sentido['hora'] = df_sentido.apply(self.ajuste_horario, axis=1, 
                                                                is_special_date=is_special_date, 
                                                                tipo_cambio_hora=tipo_cambio_hora)
                        
                        # Aggregate data
                        df_sentido = df_sentido.groupby(['Unidad de Programación', 'hora']).sum().reset_index()
                        df_sentido = df_sentido[df_sentido['volumen'] != 0.0]
                        
                        # Add date and rename columns
                        df_sentido['fecha'] = day
                        df_sentido = df_sentido.rename(columns={"Unidad de Programación": "UP"})
                        
                        # Add UP_id from dict_unidades
                        df_sentido['UP_id'] = df_sentido['UP'].map(dict_unidades)
                        
                        # Store the result
                        results.append((df_sentido, sentido))
            
            # Clean up files
            self.cleanup_files(file_name, file_name_2)
            
            return results
    

class TerciariaTransform:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def get_i90_precios(self, day: datetime) -> pd.DataFrame:
        """
        Get precios terciaria data for a specific day. 

        Note: 
            - not implemented sicne we take precios from the ESIOS API
        """
        return super().get_i90_data(day, precios_sheets = self.precios_sheets)

        pass
        

class SecundariaDL(I90DownloaderDL):
    """
    Specialized class for downloading and processing secondary regulation volume data from I90 files.
    """
    
    def __init__(self):
        """Initialize the secondary regulation downloader"""
        super().__init__()

        #initialize config
        self.config = SecundariaConfig()

        #get sheets of interest
        self.sheets_of_interest = self.config.sheets_of_interest
        
    def get_volumenes(self, day: datetime, bbdd_engine, UP_ids: Optional[List[int]] = None, 
                    sentidos: Optional[List[str]] = None) -> List[Tuple[pd.DataFrame, str]]:
        """
        Get secondary regulation volume data for a specific day.
        
        Args:
            day (datetime): Day to download data for
            bbdd_engine: Database engine connection
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            sentidos (Optional[List[str]]): List of directions to filter ['Subir', 'Bajar']
            
        Returns:
            List[Tuple[pd.DataFrame, str]]: List of tuples with (DataFrame, direction)
        """
        # Set default sentidos if None
        if sentidos is None:
            sentidos = ['Subir', 'Bajar']
        
        # Get Programming Units
        unidades, dict_unidades = self.get_programming_units(bbdd_engine, UP_ids)
        
        # Download I90 file
        file_name, file_name_2 = self.download_i90_file(day)
        
        # Get transition dates
        filtered_transition_dates = self.get_transition_dates(day, day + timedelta(days=1))
        day_datef = day.date()
        is_special_date = day_datef in filtered_transition_dates
        tipo_cambio_hora = filtered_transition_dates.get(day_datef, 0)
        
        # Get error data
        df_errores = self.get_error_data(bbdd_engine)
        df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
        pestañas_con_error = df_errores_dia['tipo_error'].values
        
        results = []
        
        # Check if the sheet has errors
        if self.sheet_id not in pestañas_con_error:
            sheet = str(self.sheet_id).zfill(2)
            
            # Load sheet data
            df = pd.read_excel(f"./I90DIA_{file_name_2}.xls", sheet_name=f'I90DIA{sheet}', skiprows=2)
            
            # Filter by programming units
            df = df[df['Unidad de Programación'].isin(unidades)]
            
            # Process for each direction
            for sentido in sentidos:
                df_sentido = df.copy()
                
                # Apply direction filter
                if sentido == 'Subir':
                    df_sentido = df_sentido[df_sentido['Sentido'] == 'Subir']
                elif sentido == 'Bajar':
                    df_sentido = df_sentido[df_sentido['Sentido'] == 'Bajar']
                
                # Filter columns to keep only programming unit and value columns
                if not df_sentido.empty:
                    total_col = df_sentido.columns.get_loc("Total")
                    cols_to_drop = list(range(0, total_col+1))
                    up_col = df_sentido.columns.get_loc("Unidad de Programación")
                    cols_to_drop.remove(up_col)
                    df_sentido = df_sentido.drop(df_sentido.columns[cols_to_drop], axis=1)
                    
                    # Melt the dataframe to convert from wide to long format
                    hora_colname = df_sentido.columns[1]
                    df_sentido = df_sentido.melt(id_vars=["Unidad de Programación"], var_name="hora", value_name="volumen")
                    
                    # Apply time adjustments
                    if hora_colname == 1:
                        df_sentido['hora'] = df_sentido.apply(self.ajuste_quinceminutal, axis=1, 
                                                             is_special_date=is_special_date, 
                                                             tipo_cambio_hora=tipo_cambio_hora)
                    else:
                        df_sentido['hora'] = df_sentido.apply(self.ajuste_horario, axis=1, 
                                                             is_special_date=is_special_date, 
                                                             tipo_cambio_hora=tipo_cambio_hora)
                    
                    # Aggregate data
                    df_sentido = df_sentido.groupby(['Unidad de Programación', 'hora']).sum().reset_index()
                    df_sentido = df_sentido[df_sentido['volumen'] != 0.0]
                    
                    # Add date and rename columns
                    df_sentido['fecha'] = day
                    df_sentido = df_sentido.rename(columns={"Unidad de Programación": "UP"})
                    
                    # Add UP_id from dict_unidades
                    df_sentido['UP_id'] = df_sentido['UP'].map(dict_unidades)
                    
                    # Store the result
                    results.append((df_sentido, sentido))
        
        # Clean up files
        self.cleanup_files(file_name, file_name_2)
        
        return results

class RRTransform:
 def get_volumenes(self, day: datetime, bbdd_engine, UP_ids: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Get RR volume data for a specific day.
        
        Args:
            day (datetime): Day to download data for
            bbdd_engine: Database engine connection
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            
        Returns:
            pd.DataFrame: DataFrame with RR volume data
        """
        # Get Programming Units
        unidades, dict_unidades = self.get_programming_units(bbdd_engine, UP_ids)
        
        # Download I90 file
        file_name, file_name_2 = self.download_i90_file(day)
        
        # Get transition dates
        filtered_transition_dates = self.get_transition_dates(day, day + timedelta(days=1))
        day_datef = day.date()
        is_special_date = day_datef in filtered_transition_dates
        tipo_cambio_hora = filtered_transition_dates.get(day_datef, 0)
        
        # Get error data
        df_errores = self.get_error_data(bbdd_engine)
        df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
        pestañas_con_error = df_errores_dia['tipo_error'].values
        
        result_df = pd.DataFrame()
        
        # Check if the sheet has errors
        if self.sheet_id not in pestañas_con_error:
            sheet = str(self.sheet_id).zfill(2)
            
            # Load sheet data
            df = pd.read_excel(f"./I90DIA_{file_name_2}.xls", sheet_name=f'I90DIA{sheet}', skiprows=2)
            
            # Filter by programming units
            df = df[df['Unidad de Programación'].isin(unidades)]
            
            # Filter for RR data - exclude indisponibilidades
            df = df[df['Redespacho'] == 'Restricciones Técnicas']
            
            # Filter columns to keep only programming unit and value columns
            if not df.empty:
                total_col = df.columns.get_loc("Total")
                cols_to_drop = list(range(0, total_col+1))
                up_col = df.columns.get_loc("Unidad de Programación")
                cols_to_drop.remove(up_col)
                df = df.drop(df.columns[cols_to_drop], axis=1)
                
                # Melt the dataframe to convert from wide to long format
                hora_colname = df.columns[1]
                df = df.melt(id_vars=["Unidad de Programación"], var_name="hora", value_name="volumen")
                
                # Apply time adjustments
                if hora_colname == 1:
                    df['hora'] = df.apply(self.ajuste_quinceminutal, axis=1, 
                                         is_special_date=is_special_date, 
                                         tipo_cambio_hora=tipo_cambio_hora)
                else:
                    df['hora'] = df.apply(self.ajuste_horario, axis=1, 
                                         is_special_date=is_special_date, 
                                         tipo_cambio_hora=tipo_cambio_hora)
                
                # Aggregate data
                df = df.groupby(['Unidad de Programación', 'hora']).sum().reset_index()
                df = df[df['volumen'] != 0.0]
                
                # Add date and rename columns
                df['fecha'] = day
                df = df.rename(columns={"Unidad de Programación": "UP"})
                
                # Add UP_id from dict_unidades
                df['UP_id'] = df['UP'].map(dict_unidades)
                
                result_df = df
        
        # Clean up files
        self.cleanup_files(file_name, file_name_2)
        
        return result_df
 
 class CurtailmentTransform:
    def __init__(self):
        pass

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def get_volumenes(self, day: datetime, bbdd_engine, UP_ids: Optional[List[int]] = None,
                     tipos: Optional[List[str]] = None) -> List[Tuple[pd.DataFrame, str]]:
        """
        Get curtailment volume data for a specific day.
        
        Args:
            day (datetime): Day to download data for
            bbdd_engine: Database engine connection
            UP_ids (Optional[List[int]]): List of programming unit IDs to filter
            tipos (Optional[List[str]]): List of curtailment types to filter ['Curtailment', 'Curtailment demanda']
            
        Returns:
            List[Tuple[pd.DataFrame, str]]: List of tuples with (DataFrame, curtailment type)
        """
        # Set default tipos if None
        if tipos is None:
            tipos = ['Curtailment', 'Curtailment demanda']
        
        # Get Programming Units
        unidades, dict_unidades = self.get_programming_units(bbdd_engine, UP_ids)
        
        # Download I90 file
        file_name, file_name_2 = self.download_i90_file(day)
        
        # Get transition dates
        filtered_transition_dates = self.get_transition_dates(day, day + timedelta(days=1))
        day_datef = day.date()
        is_special_date = day_datef in filtered_transition_dates
        tipo_cambio_hora = filtered_transition_dates.get(day_datef, 0)
        
        # Get error data
        df_errores = self.get_error_data(bbdd_engine)
        df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
        pestañas_con_error = df_errores_dia['tipo_error'].values
        
        results = []
        
        # Check if the sheet has errors
        if self.sheet_id not in pestañas_con_error:
            sheet = str(self.sheet_id).zfill(2)
            
            # Load sheet data
            df = pd.read_excel(f"./I90DIA_{file_name_2}.xls", sheet_name=f'I90DIA{sheet}', skiprows=2)
            
            # Filter by programming units
            df = df[df['Unidad de Programación'].isin(unidades)]
            
            # Process for each tipo
            for tipo in tipos:
                df_tipo = df.copy()
                
                # Apply curtailment type filter
                if tipo == 'Curtailment':
                    df_tipo = df_tipo[df_tipo['Redespacho'].isin(['UPLPVPV', 'UPLPVPCBN'])]
                elif tipo == 'Curtailment demanda':
                    df_tipo = df_tipo[df_tipo['Redespacho'] == 'UPOPVPB']
                
                # Filter columns to keep only programming unit and value columns
                if not df_tipo.empty:
                    total_col = df_tipo.columns.get_loc("Total")
                    cols_to_drop = list(range(0, total_col+1))
                    up_col = df_tipo.columns.get_loc("Unidad de Programación")
                    cols_to_drop.remove(up_col)
                    df_tipo = df_tipo.drop(df_tipo.columns[cols_to_drop], axis=1)
                    
                    # Melt the dataframe to convert from wide to long format
                    hora_colname = df_tipo.columns[1]
                    df_tipo = df_tipo.melt(id_vars=["Unidad de Programación"], var_name="hora", value_name="volumen")
                    
                    # Apply time adjustments
                    if hora_colname == 1:
                        df_tipo['hora'] = df_tipo.apply(self.ajuste_quinceminutal, axis=1, 
                                                       is_special_date=is_special_date, 
                                                       tipo_cambio_hora=tipo_cambio_hora)
                    else:
                        df_tipo['hora'] = df_tipo.apply(self.ajuste_horario, axis=1, 
                                                       is_special_date=is_special_date, 
                                                       tipo_cambio_hora=tipo_cambio_hora)
                    
                    # Aggregate data
                    df_tipo = df_tipo.groupby(['Unidad de Programación', 'hora']).sum().reset_index()
                    df_tipo = df_tipo[df_tipo['volumen'] != 0.0]
                    
                    # Add date and rename columns
                    df_tipo['fecha'] = day
                    df_tipo = df_tipo.rename(columns={"Unidad de Programación": "UP"})
                    
                    # Add UP_id from dict_unidades
                    df_tipo['UP_id'] = df_tipo['UP'].map(dict_unidades)
                    
                    # Store the result
                    results.append((df_tipo, tipo))
        
        # Clean up files
        self.cleanup_files(file_name, file_name_2)
        
        return results