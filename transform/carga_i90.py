import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime,date,timedelta, time
import math
import pytz
import requests
import sqlalchemy
import zipfile
import os
import time


################################################################   ALGORITMO DE CARGA DE DATOS DEL i90   ################################################################

def algo_i90(esios_token,bbdd_engine,fecha_inicio_carga=None,fecha_fin_carga=None,UP_ids=None,mercados_ids=None):

    #Definimos los días (y por tanto los ficheros i90) para los que queremos cargar datos
    if not fecha_inicio_carga:
        fecha_inicio_carga_datef = date.today() + timedelta(-92)
        fecha_inicio_carga = fecha_inicio_carga_datef.strftime("%m/%d/%Y")
    fecha_inicio_carga_datef = datetime.strptime(fecha_inicio_carga, "%m/%d/%Y")

    if not fecha_fin_carga:
        fecha_fin_carga_datef = date.today() + timedelta(-92)
        fecha_fin_carga = fecha_fin_carga_datef.strftime("%m/%d/%Y")
    fecha_fin_carga_datef = datetime.strptime(fecha_fin_carga, "%m/%d/%Y") + timedelta(days=1)
    print(fecha_inicio_carga_datef,fecha_fin_carga_datef)
    
    days = pd.date_range(start=fecha_inicio_carga, end=fecha_fin_carga)

    #Cargamos los días de cambio de hora (con 23 y 25 horas) que hay dentro del rango de carga
    spain_timezone = pytz.timezone('Europe/Madrid')
    utc_transition_times = spain_timezone._utc_transition_times[1:]
    localized_transition_times = [pytz.utc.localize(transition).astimezone(spain_timezone) for transition in utc_transition_times]
    fecha_inicio_local = spain_timezone.localize(fecha_inicio_carga_datef)
    fecha_fin_local = spain_timezone.localize(fecha_fin_carga_datef)
    filtered_transition_times = [transition for transition in localized_transition_times if fecha_inicio_local <= transition <= fecha_fin_local]
    filtered_transition_dates = {dt.date():int(dt.isoformat()[-4]) for dt in filtered_transition_times}
    #print(filtered_transition_dates)

    #Cargamos las Unidades de Programación de interés de la base de datos
    query = '''SELECT u.id as id,UP FROM UPs u inner join Activos a on u.activo_id = a.id where a.region = "ES"'''
    if UP_ids:
        UP_list = """, """.join([str(item) for item in UP_ids])
        query += f' and u.id in ({UP_list})'
    df_up = pd.read_sql_query(query,con=bbdd_engine)
    unidades = df_up['UP']
    dict_unidades = dict(zip(unidades, df_up['id']))

    #Descargamos de la BBDD los mercados de los que queremos guardar datos
    query = '''SELECT * FROM Mercados
            where sheet_i90_volumenes != 0'''
    if mercados_ids:
        mercados_list = """, """.join([str(item) for item in mercados_ids])
        query += f' and id in ({mercados_list})'
    df_mercados = pd.read_sql_query(query,con=bbdd_engine)
    pestañas_volumenes = df_mercados['sheet_i90_volumenes'].unique().tolist()
    pestañas = pestañas_volumenes + df_mercados['sheet_i90_precios'].unique().tolist()
    pestañas = [int(item) for item in pestañas if item is not None and not (isinstance(item, float) and math.isnan(item))]

    #Descargamos de la BBDD las pestañas sin datos por errores o sesiones canceladas
    query = '''SELECT fecha,tipo_error FROM Errores_i90_OMIE where fuente_error = "i90"'''
    df_errores = pd.read_sql_query(query,con=bbdd_engine)
    df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date

    #Definimos fechas de cambios regulatorios
    dia_inicio_SRS = datetime(2024,11,20)

    #Iteramos a lo largo de los días que queremos cargar. Cada día corresponde a un fichero i90
    for day in tqdm(days):

        #Descargamos el zip con el fichero i90
        address = "https://api.esios.ree.es/archives/34/download?date_type\u003ddatos\u0026end_date\u003d" + str(day.date()) + "T23%3A59%3A59%2B00%3A00\u0026locale\u003des\u0026start_date\u003d" + str(day.date()) + "T00%3A00%3A00%2B00%3A00"

        resp = requests.get(address, headers={'x-api-key' : '%s' %(esios_token),
                                                'Authorization' : 'Token token=%s' %(esios_token)})
        
        with open( str(day.date()) + ".zip", "wb") as fd:
            fd.write(resp.content)

        file_name = str(day.year) + "-" + str(day.month).zfill(2) + "-" + str(day.day).zfill(2)
        file_name_2 = str(day.year) + str(day.month).zfill(2) + str(day.day).zfill(2)

        #Descomprimimos el fichero zip
        with zipfile.ZipFile("./" + file_name + ".zip", 'r') as zip_ref:
            zip_ref.extractall("./")

        #Comprobamos si se trata de un día de 23 o 25 horas para los ajustes horarios posteriores
        day_datef = day.date()
        is_special_date = day_datef in filtered_transition_dates
        tipo_cambio_hora = filtered_transition_dates.get(day_datef,0)
        #print("")
        #print(day_datef,is_special_date,tipo_cambio_hora)

        #Filtramos los errores si los hubiera del día en cuestión
        df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
        pestañas_con_error = df_errores_dia['tipo_error'].values
        
        #Leemos los datos del excel (i90)
        for pestaña in pestañas: 
            if pestaña not in pestañas_con_error and (pestaña != 5 or day < dia_inicio_SRS): 
                #Definimos la pestaña en formato y cargamos los datos
                sheet = str(pestaña).zfill(2) 
                if sheet not in ['03','05','06','07','08','09','10']:
                    df = pd.read_excel("./I90DIA_" + file_name_2 + ".xls", sheet_name='I90DIA'+sheet, skiprows=3)
                else:
                    df = pd.read_excel("./I90DIA_" + file_name_2 + ".xls", sheet_name='I90DIA'+sheet, skiprows=2)

                if sheet == '05' and 'Participante del Mercado' in df.columns:
                    df = df.rename(columns={'Participante del Mercado': 'Unidad de Programación'})
                
                #Filtramos por las unidades de programación de interés
                df = df[df['Unidad de Programación'].isin(unidades)]

                #Definimos los mercados a cargar dentro de esa pestaña
                if pestaña in pestañas_volumenes:
                    is_precio = False
                    df_mercados_filtrado = df_mercados[df_mercados['sheet_i90_volumenes'] == pestaña]
                else:
                    is_precio = True
                    df_mercados_filtrado = df_mercados[df_mercados['sheet_i90_precios'] == pestaña]

                df_sheet = df.copy()

                #Iteramos sobre dichos mercados
                for index,mercado in df_mercados_filtrado.iterrows(): 

                    print("")
                    print("\033[20;1m" + mercado['mercado'].upper() + "\033[0m")
                    print("")

                    df = df_sheet.copy()

                    #Realizamos el filtro de ofertas según corresponda
                    if mercado['sentido'] == 'Subir':
                        #print('Filtro a subir')
                        df = df[df['Sentido'] == 'Subir']
                    elif mercado['sentido'] == 'Bajar':
                        #print('Filtro a bajar')
                        df = df[df['Sentido'] == 'Bajar']
                    #else:
                        #print("Sin filtro de sentido")

                    if sheet == '03':
                        #Apply redespacho filter for Curtialment class 
                        if mercado['mercado'] in ['Curtailment','Curtailment demanda']:
                            df = df[df['Redespacho'].isin(['UPLPVPV','UPLPVPCBN','UPOPVPB'])]

                       #filter for restricciones class specifically for rt2 
                        elif mercado['mercado'] in ['RT2 a subir','RT2 a bajar']:
                            df = df[df['Redespacho'].isin(['ECOBSO','ECOBCBSO'])]
                            
                        #filter for restricciones class specifically for md 
                        else:
                            df = df[df['Redespacho'].isin(['ECO','ECOCB','UPOPVPV','UPOPVPVCB'])]

                    elif sheet == '07':
                        #no filter here for terciaria 
                        if mercado['mercado'] in ['Terciaria a subir','Terciaria a bajar']:
                            df = df[df['Redespacho'] != 'TERDIR']
                        else: #terciaria programada
                            df = df[df['Redespacho'] == 'TERDIR']
                    
                    elif sheet == '08' or sheet == '10':
                        #Apply redespacho filter for Indisponibilidades class in volumenes only for sheet 8 
                        if mercado['mercado'] == "Indisponibilidades":
                            df = df[df['Redespacho'] == "Indisponibilidad"]
                        else:
                            #apply redespacho filter for Restricciones Técnicas class in volumenes for sheet 8 and sheet 10
                            #print("Filtro de Restricciones Técnicas")
                            df = df[df['Redespacho'] == "Restricciones Técnicas"]

                    elif sheet == '09': #restricciones md precios 
                        df = df[df['Redespacho'].isin(['ECO','ECOCB','UPOPVPV','UPOPVPVCB'])]

                    #Filtramos las columnas que nos interesan (UP + valores)
                    total_col = df.columns.get_loc("Total")
                    cols_to_drop = list(range(0,total_col+1))
                    up_col = df.columns.get_loc("Unidad de Programación")
                    cols_to_drop.remove(up_col)
                    df = df.drop(df.columns[cols_to_drop], axis=1)
                    
                    #Aplicamos los procesos de conversión horaria necesarios y agregamos los valores en caso de ser necesario
                    hora_colname = df.columns[1]
                    df = df.melt(id_vars=["Unidad de Programación"], var_name="hora", value_name="valor")

                    if len(df) > 0:
                        day_datef = day.date()
                        is_special_date = day_datef in filtered_transition_dates
                        tipo_cambio_hora = filtered_transition_dates.get(day_datef,0)
                        if hora_colname == 1:
                            if mercado['is_quinceminutal']:
                                df['hora'] = df.apply(ajuste_quinceminutal,axis=1,is_special_date=is_special_date,tipo_cambio_hora=tipo_cambio_hora)
                            else:
                                df['hora'] = df.apply(ajuste_quinceminutal_a_horario,axis=1,is_special_date=is_special_date,tipo_cambio_hora=tipo_cambio_hora)
                        else:
                            df['hora'] = df.apply(ajuste_horario,axis=1,is_special_date=is_special_date,tipo_cambio_hora=tipo_cambio_hora)       
                        
                        if is_precio:
                            #print("Ponderado de precio")
                            df = df.groupby(['Unidad de Programación','hora']).mean().reset_index()
                            df['valor'] = df['valor'].round(decimals=3)
                        else:
                            df = df.groupby(['Unidad de Programación','hora']).sum().reset_index()
                            df = df[df['valor'] != 0.0]
                    
                    #Completamos los datos que faltan y ajustamos al formato de la BBDD 
                    df['fecha'] = day
                    df['id_mercado'] = mercado['id']
                    df = df.rename(columns={"Unidad de Programación": "UP"})
                    df = df.merge(df_up, how='left', left_on="UP", right_on="UP")
                    df.rename(columns={"id": "UP_id"}, inplace=True)
                    df.drop(columns=["UP"], inplace=True)
                    df = df[df['valor'].notna()]
                        
                    print("")
                    print(df)
                    print("")       
                    print("")

                    if is_precio:
                        df.rename(columns={"valor": "precio"}, inplace=True)
                        df.to_sql("Precios_UP",con=bbdd_engine,if_exists='append',index=False)
                        print('Carga horaria realizada correctamente')
                    else:
                        df.rename(columns={"valor": "volumen"}, inplace=True)
                        if hora_colname == 1 and mercado['is_quinceminutal']:
                            df.to_sql("Volumenes_quinceminutales",con=bbdd_engine,if_exists='append',index=False)
                            print('Carga quinceminutal realizada correctamente')
                        else:
                            df.to_sql("Volumenes_horarios",con=bbdd_engine,if_exists='append',index=False)
                            print('Carga horaria realizada correctamente')
            

        time.sleep(5)

        os.remove("./" + file_name + ".zip")
        os.remove("./I90DIA_" + file_name_2 + ".xls")

    return


################################################################   SUBFUNCIONES DEL ALGORITMO   ################################################################

#Funciones de ajuste horario
def ajuste_horario(row,is_special_date,tipo_cambio_hora):
    if is_special_date:
        if tipo_cambio_hora == 2:  #Dia 23 horas
            if int(row['hora'][-2:]) < 3:
                row['hora'] = int(row['hora'][-2:])
            else:
                row['hora'] = int(row['hora'][-2:]) - 1
        
        if tipo_cambio_hora == 1:  #Dia 25 horas
            if row['hora'][-1].isdigit():
                if int(row['hora'][-2:]) < 3:
                    row['hora'] = int(row['hora'][-2:])
                else:
                    row['hora'] = int(row['hora'][-2:]) + 1
            elif row['hora'][-1] == 'a':
                row['hora'] = int(row['hora'][-3:-1])
            elif row['hora'][-1] == 'b':
                row['hora'] = int(row['hora'][-3:-1]) + 1
    
    else:    #Resto de días
        row['hora'] = int(row['hora'][-2:])

    return row['hora']

def ajuste_quinceminutal(row,is_special_date,tipo_cambio_hora):
    minutos_dict = {0:":00",1:":15",2:":30",3:":45"}
    if is_special_date and tipo_cambio_hora == 2:   #Dia de 23 horas
        if row['hora'] > 8:
            hora = str((row['hora']+3)//4 - 2).zfill(2)
        else:
            hora = str((row['hora']+3)//4 - 1).zfill(2)

    else:              #Dias normales y con 25 horas que no requieren ajuste especial
        hora = str((row['hora']+3)//4 - 1).zfill(2)
        
    minutos = minutos_dict[(row['hora']+3)%4]
    row['hora'] = hora + minutos

    return row['hora']

def ajuste_quinceminutal_a_horario(row,is_special_date,tipo_cambio_hora):
    if is_special_date and tipo_cambio_hora == 2:   #Dia de 23 horas
        if row['hora'] > 8:
            row['hora'] = (row['hora']+3)//4 - 1
        else:
            row['hora'] = (row['hora']+3)//4
    
    else:              #Dias normales y con 25 horas que no requieren ajuste especial
        row['hora'] = (row['hora']+3)//4

    return row['hora']