import pandas as pd
from tqdm import tqdm
import numpy as np
from datetime import datetime,date,timedelta
import pytz
import requests
import sqlalchemy
import zipfile
import os

################################################################   ALGORITMO DE CARGA DE DATOS DE LOS ARCHIVOS DE OMIE   ################################################################

def algo_omie(bbdd_engine,fecha_inicio_carga=None,fecha_fin_carga=None,UP_ids=None,cargar_intras=None,cargar_continuo=True):

    #Definimos los días (y por tanto los ficheros i90) para los que queremos cargar datos
    if fecha_inicio_carga is None:
        fecha_inicio_carga_datef = date.today() + timedelta(-92)
        fecha_inicio_carga = fecha_inicio_carga_datef.strftime("%m/%d/%Y")
    
    fecha_inicio_carga_datef = datetime.strptime(fecha_inicio_carga, "%m/%d/%Y")

    if fecha_fin_carga is None:
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

    #Cargamos las Unidades de Oferta de interés de la base de datos
    query = '''SELECT u.id as id,UP,UOF FROM UPs u inner join Activos a on u.activo_id = a.id where a.region = "ES"'''
    if UP_ids:
        UP_list = """, """.join([str(item) for item in UP_ids])
        query += f' and u.id in ({UP_list})'
    df_up = pd.read_sql_query(query,con=bbdd_engine)
    
    unidades = df_up['UOF']
    dict_unidades = dict(zip(unidades, df_up['id'])) 

    #Descargamos de la BBDD las pestañas sin datos por errores o sesiones canceladas
    query = '''SELECT fecha,tipo_error FROM Errores_i90_OMIE where fuente_error = "omie-intra"'''
    df_errores = pd.read_sql_query(query,con=bbdd_engine)
    df_errores['fecha'] = pd.to_datetime(df_errores['fecha']).dt.date

    mes_ref = None
    año_ref = None
    #Iteramos a lo largo de los días que queremos cargar. Cada día corresponde a un fichero i90
    for day in tqdm(days):
        
        #Filtramos los errores si los hubiera del día en cuestión
        df_errores_dia = df_errores[df_errores['fecha'] == day.date()]
        sesiones_canceladas = df_errores_dia['tipo_error'].values.tolist()

        #Si hay cambio en las UOF
        #df_up.apply(lambda row: row['UOF'] if day >= pd.to_datetime(row['fecha_cambio']) else row['UOF_old'], axis=1)

        # Carga de Intras
        if cargar_intras is not False:
            
            if año_ref != day.year or mes_ref != day.month:
                #Eliminamos el anterior
                if año_ref:
                    os.remove("./" + file_name_intras + ".zip")
                #Descargamos el zip
                file_name_intras = f"curva_pibc_uof_{day.year}{str(day.month).zfill(2)}"
                address = f"https://www.omie.es/es/file-download?parents=curva_pibc_uof&filename={file_name_intras}.zip"
                resp = requests.get(address)   
                with open( file_name_intras + ".zip", "wb") as fd:
                    fd.write(resp.content)

            #sesiones de los intradiarios habilitadas
            if cargar_intras is None or cargar_intras is True: 
                intradiarios = [1,2,3,4,5,6]
            else:
                intradiarios = cargar_intras
            intradiarios = [str(item).zfill(2) for item in intradiarios if item not in sesiones_canceladas]
            
            #Cargamos los datos de la energía casada en los intradiarios
            for intra in intradiarios:

                if intra not in ['04','05','06'] or day.date() <= date(2024,6,13):
                    
                    session_file_name = file_name_intras + str(day.day).zfill(2) + intra + ".1"
                    #Descomprimimos el fichero zip
                    with zipfile.ZipFile("./" + file_name_intras + ".zip", 'r') as zip_ref:
                        zip_ref.extract(session_file_name, "./")


                    df = pd.read_csv("./"+session_file_name, sep=";", skiprows=2, encoding='latin-1')
                    
                    df = df[df['Unidad'].isin(unidades)]
                    df = df.replace(dict_unidades)
                    df = df.rename(columns={"Unidad":"UP_id"})


                    df = df[df['Ofertada (O)/Casada (C)'] == 'C']
                    df['Extra'] = np.where(df['Tipo Oferta']== 'C', -1, 1)
                    #print(df)
                    df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace('.', '')
                    df['Energía Compra/Venta'] = df['Energía Compra/Venta'].str.replace(',', '.')
                    df['Energía Compra/Venta'] = df['Energía Compra/Venta'].astype(float)
                    df['Energía Compra/Venta'] = df['Energía Compra/Venta'] * df['Extra']
                    df['Hora'] = df['Hora'].astype(int)
                    df['Fecha'] = pd.to_datetime(df['Fecha'],format="%d/%m/%Y")
                    df = df.groupby(['UP_id','Fecha','Hora']).sum().reset_index()
                    if intra == '02':
                        df['id_mercado'] = np.where(df['Fecha']==day,3,8)
                    else:
                        df['id_mercado'] = int(intra) + 1
                    df = df.rename(columns = {'Energía Compra/Venta':'Volumen'})
                    df = df[['Fecha','Hora','UP_id','Volumen','id_mercado']]

                    print(df)

                    df.to_sql("Volumenes_horarios",con=bbdd_engine,if_exists='append',index=False)

                    os.remove("./" + session_file_name)

            
        # Cargamos el continuo
        if cargar_continuo:
        
            if año_ref != day.year or mes_ref != day.month:
                #Eliminamos el anterior
                if año_ref:
                    os.remove("./" + file_name_mic + ".zip")
                file_name_mic = f"trades_{day.year}{str(day.month).zfill(2)}"
                #Descargamos el zip con el fichero i90
                address = f"https://www.omie.es/es/file-download?parents=trades&filename={file_name_mic}.zip"
                resp = requests.get(address)
                
                with open( file_name_mic + ".zip", "wb") as fd:
                    fd.write(resp.content)

            #Descomprimimos el fichero zip
            day_file_name = file_name_mic + str(day.day).zfill(2) + ".1"
            with zipfile.ZipFile("./" + file_name_mic + ".zip", 'r') as zip_ref:
                zip_ref.extract(day_file_name, "./")

            mic = pd.read_csv("./"+day_file_name, sep=";", skiprows=2, encoding='latin-1')
            df = mic[['Fecha', 'Contrato', 'Unidad compra', 'Unidad venta', 'Precio', 'Cantidad']]

            df_compra = df[df['Unidad compra'].isin(unidades)]
            df_compra.loc[:, 'Cantidad'] = df_compra['Cantidad'].str.replace('.', '')
            df_compra.loc[:, 'Cantidad'] = df_compra['Cantidad'].str.replace(',', '.')
            df_compra.loc[:, 'Cantidad'] = -(df_compra['Cantidad'].astype(float))
            df_compra = df_compra.rename(columns={"Unidad compra":"UP_id"})
            df_compra = df_compra[['Fecha','Contrato','UP_id','Precio','Cantidad']]

            df_venta  = df[df['Unidad venta' ].isin(unidades)]
            df_venta.loc[:, 'Cantidad'] = df_venta['Cantidad'].str.replace('.', '')
            df_venta.loc[:, 'Cantidad'] = df_venta['Cantidad'].str.replace(',', '.')
            df_venta.loc[:, 'Cantidad'] = df_venta['Cantidad'].astype(float)
            df_venta = df_venta.rename(columns={"Unidad venta": "UP_id"})
            df_venta = df_venta[['Fecha','Contrato','UP_id','Precio','Cantidad']]


            df = pd.concat([df_compra,df_venta])

            if len(df) > 0:
                df = df.replace(dict_unidades)
            
                df['Fecha'] = pd.to_datetime(df['Fecha'],format="%d/%m/%Y")
                df['Fecha'] = df['Fecha'].dt.strftime('%Y-%m-%d')
                df = df.rename(columns={"Fecha":"Fecha_fichero"})
                
                df['Precio'] = df['Precio'].str.replace('.', '')
                df['Precio'] = df['Precio'].str.replace(',', '.')
                df['Precio'] = df['Precio'].astype(float)

                df['Fecha'] = df['Contrato'].str.strip().str[:8]
                df['Fecha'] = pd.to_datetime(df['Fecha'],format="%Y%m%d")
                df['Fecha'] = df['Fecha'].dt.strftime('%Y-%m-%d')
                
                df['Hora'] = df.apply(ajuste_horario,axis=1,args=(filtered_transition_dates,))

                df = df.rename(columns={'Cantidad':'volumen'})

                df = df[['Fecha','Hora','UP_id','volumen','Precio','Fecha_fichero']]
                
                print(df)

                df.to_sql("Operaciones_MIC",con=bbdd_engine,if_exists='append',index=False)
                
            os.remove("./" + day_file_name)
        
        mes_ref = day.month
        año_ref = day.year
    
    if cargar_intras is not False:
        os.remove("./" + file_name_intras + ".zip")
    if cargar_continuo:
        os.remove("./" + file_name_mic + ".zip")

    return


################################################################   SUBFUNCIONES DEL ALGORITMO   ################################################################

#Funciones de ajuste horario
def ajuste_horario(row,filtered_transition_dates):#,is_special_date,tipo_cambio_hora):
    date_ref = datetime.strptime(row['Fecha'], '%Y-%m-%d').date()
    is_special_date = date_ref in filtered_transition_dates
    tipo_cambio_hora = filtered_transition_dates.get(date_ref,0)
    
    if is_special_date:
        if tipo_cambio_hora == 2:  #Dia 23 horas
            if (int(row['Contrato'].strip()[9:11]) + 1) < 3:
                row['Hora'] = int(row['Contrato'].strip()[9:11]) + 1
            else:
                row['Hora'] = int(row['Contrato'].strip()[9:11])
        
        if tipo_cambio_hora == 1:  #Dia 25 horas
            if row['Contrato'].strip()[-1].isdigit():
                if (int(row['Contrato'].strip()[9:11]) + 1) < 3:
                    row['Hora'] = int(row['Contrato'].strip()[9:11]) + 1
                else:
                    row['Hora'] = int(row['Contrato'].strip()[9:11]) + 2
            elif row['Contrato'].strip()[-1] == 'A':
                row['Hora'] = int(row['Contrato'].strip()[9:11]) + 1
            elif row['Contrato'].strip()[-1] == 'B':
                row['Hora'] = int(row['Contrato'].strip()[9:11]) + 2
            else:
                print("Error")
    
    else:    #Resto de días
        row['Hora'] = int(row['Contrato'].strip()[9:11]) + 1

    return row['Hora']
