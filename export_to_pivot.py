
# coding: utf-8

# In[1]:

'''
Cria uma função que transforma o .csv de histórico de preços ('price export') padronizado
(pelo Geron) em um pandas.DataFrame em formato de tabela dinâmica, apropriado para o cálculo do IPC-W.
'''

import pandas as pd
import numpy as np
from datetime import timedelta


def dias_faltantes(price_pivot):
    """
    Gera uma lista com as datas faltantes na pivot_table (durante todo o periodo)
    """
    a = price_pivot.columns[6:]
    mydates = pd.to_datetime(a)
    date1 = a[0]
    date2 = a[-1]
    alldates = pd.date_range(date1, date2)
    datas_faltantes = [ item.strftime('%Y-%m-%d') for item in alldates if item not in mydates]
    
    return datas_faltantes

def fill_datas_faltantes(price_pivot, datas_faltantes):
    '''Adiciona colunas com NAN para as datas faltantes no pivot_price resultante do export_to_pivot 
    e retorna o novo pivot_price'''
    for data in datas_faltantes:
        data_anterior = (pd.to_datetime(data) - timedelta(days=1))
        idx_data_anterior = price_pivot.columns.get_loc(data_anterior.strftime('%Y-%m-%d'))
        idx_data_faltante = idx_data_anterior + 1
        new_col = np.empty(len(price_pivot))
        new_col[:] = np.NAN
        data_string = (data_anterior + timedelta(days=1)).strftime('%Y-%m-%d')
        price_pivot.insert(loc=idx_data_faltante, column=data_string, value=new_col)
    return price_pivot

def export_to_pivot(price_export_csv_file, drop_regular_prices = False, intraday_aggfunc = 'mean'):
    '''
    Transforma o .csv de histórico de preços em um pandas.DataFrame ajustado para o cálculo do IPC-W.
    
    Parâmetros
    ----------
    price_export_csv_file : str
        Caminho para o arquivo .csv padronizado, criado pelo Geron ('price export' ou 'histórico de preços').
    
    drop_regular_prices : bool, default False
        Se False, a informação da coluna 'regular_price' é aproveitada
        nos casos em que a coluna 'price' não possui dados.
        Se True, os dados da coluna 'regular_price' são descartados
        nos casos em que a coluna 'price' não possui dados.
    
    intraday_aggfunc : {'last', 'first', 'mean', 'median', 'min', 'max'}, default 'mean'
        Define qual tratamento dar para os casos de mais de um preço coletado no mesmo dia.
        Valores missing são desprezados nesta agregação, para qualquer valor do parâmetro.
        - 'last' : Considera apenas o último preço coletado.
        - 'first' : Considera apenas o primeiro preço coletado.
        - 'mean' : Calcula a média dos preços coletados.
        - 'median' : Calcula a mediana dos preços coletados.
        - 'min' : Considera apenas o menor preço coletado.
        - 'max' : Considera apenas o maior preço coletado.
        
    Resultado
    ---------
    price_pivot : pandas.DataFrame
        O output fica em forma de tabela dinâmica, onde cada linha representa um insumo distinto.
        Além das informações do .csv original, cada dia de coleta fica em uma coluna própria do output.
        
    datas_faltando: list
        Lista com as datas que não existem dentro do período entre o dia inicial e final do price_pivot
    '''
    
    price_export = pd.read_csv(price_export_csv_file, dtype = {'website_id': str, 'sku': str}).drop("product_name",axis = 1)
    #dropa product_name pois já temos global_product_name
    
    if drop_regular_prices:
        is_dropped_price = pd.isnull(price_export.price) & pd.notnull(price_export.regular_price)
        price_export.loc[is_dropped_price, ['regular_price']] = np.nan
        print('')
        print('Atenção: foram descartados', str(sum(is_dropped_price)), 'preços.')
        print('')
    
    price_export.price = price_export.loc[:, ['price', 'regular_price']].min(axis = 1, skipna = True)
    price_export.drop(['regular_price'], axis = 1, inplace = True)
    
    price_export['day'] = price_export.visit_date.apply(lambda x: x.split()[0])
    duplicated_observations = price_export.duplicated(subset = ['website_id', 'sku', 'day'], keep = False)
    price_is_null = pd.isnull(price_export.price)
    price_export = price_export[~(duplicated_observations & price_is_null)]
    
    if intraday_aggfunc in {'first', 'last'}:
        price_export.visit_date = pd.to_datetime(price_export.visit_date)
        price_export.sort_values('visit_date', inplace = True, kind = 'mergesort')
        still_duplicated_observations = price_export.duplicated(subset = ['website_id', 'sku', 'day'],
                                                                keep = intraday_aggfunc)
        price_export = price_export[~still_duplicated_observations]
        intraday_aggfunc = 'mean'
    
    price_export.drop(['visit_date'], axis = 1, inplace = True)
    price_export['insumo_id'] = price_export.website_id + '_-_' + price_export.sku
    
    price_pivot = pd.pivot_table(price_export, index = ['insumo_id', 'website_id', 'website_name', 'sku',
                                                        'global_product_name', 'ipc_subitem_name', 'ipc_subitem_id'],
                                 columns = 'day', values = 'price', aggfunc = intraday_aggfunc)
    price_pivot.reset_index(inplace = True)
    price_pivot.drop(['insumo_id'], axis = 1, inplace = True)
    price_pivot.columns.name = None
    
    return price_pivot
