
# coding: utf-8

# In[26]:

import export_to_pivot as exp
import filtro_dias as fd
import pandas as pd

DATE_FORMAT = '%Y-%m-%d' # See http://strftime.org/
DATE_STEP = pd.Timedelta(days=1)

def gera_pivot_only_date_cols(price_pivot):
    '''
    Elimina primeiras 6 colunas da price_pivot (colunas que não são referentes a preço), deixando apenas as de datas
    (que terão os valores dos preços dos produtos), passando-as pro formato datetime.
    '''
    ppivot_only_date_cols = price_pivot.copy()
    ppivot_only_date_cols = ppivot_only_date_cols[ppivot_only_date_cols.columns[6:]]
    ppivot_only_date_cols.columns = pd.to_datetime(ppivot_only_date_cols.columns, format= "%Y-%m-%d")
    return ppivot_only_date_cols

def remove_time_from_datetime_cols(ppivot_only_date_cols):
    '''
    Remove do time (hora) dos títulos das colunas (deixando apenas a data)
    '''
    ppivot_only_date_cols.columns = [ppivot_only_date_cols.columns[i].date() for i in range(ppivot_only_date_cols.shape[1])]
    ppivot_date_cols_no_time = ppivot_only_date_cols
    return ppivot_date_cols_no_time

def retira_primeiras_cols_converte_datetime(price_pivot):
    '''
    Junção de gera_pivot_only_date_col() + remove_time_from_datetime_cols()
    
    1.Elimina primeiras 6 colunas da price_pivot (colunas que não são referentes a preço), deixando apenas as de datas
    (que terão os valores dos preços dos produtos), passando-as pro formato datetime.
    
    2.Remove do time (hora) dos títulos das colunas (deixando apenas a data)
    '''
    ppivot_only_date_cols = gera_pivot_only_date_cols(price_pivot)
    ppivot_date_cols_no_time = remove_time_from_datetime_cols(ppivot_only_date_cols)
    
    return ppivot_date_cols_no_time

def strftime(date):
    return date.strftime(DATE_FORMAT)

def str_range_params_to_datetime(start, end, span_days):
    '''
    Pega os parametros (string) que vão ser utilizados na intervals_date_range_generator e passa para datetime
    '''
    try:
        start = pd.to_datetime(start, format=DATE_FORMAT)
    except ValueError:
        print        ("Ooops! Time format inválido. Coloque datas no formato Y-m-d (Ano *com 4 digitos*, mês *em algarismos*, dia)")
    try:
        end = pd.to_datetime(end, format=DATE_FORMAT)  
    except ValueError:
        print        ("Ooops! Time format inválido. Coloque datas no formato Y-m-d (Ano *com 4 digitos*, mês *em algarismos*, dia)")     
    span  = pd.Timedelta(days=span_days)
    return start, end, span


def intervals_date_range_generator(start, end, span_days, consider_diff_month_intervals = 0):
    """
    Gera uma lista com intervalos de tamanho fixo dentro das datas (periodo) desejadas.
    (menos o ultimo que pode ser menor)
    
    Entradas:
        start: data inicial do periodo utilizado
        end: data final do periodo utilizado
        span_days: tamanho dos intervalos desejados no periodo
        consider_diff_months_intervals: flag que quando setada com qualquer valor diferente de 0 irá considerar
            intervalos que comecem em um mês e terminem em outro. Por DEFAULT elimina esse tipo de intervalo
    Saída: 
        Lista com todos os intervalos daquele tamanho. 
        CAUTION: Ultimo intervalo pode não ter o tamanho desejado (span_days)
    
    Ex de uso (sem flag):
    intervals_date_range_generator('2017-01-01', '2017-02-07', 7)

    1st element = ('2017-01-01', '2017-02-07')
    2nd element = ('2017-01-08', '2017-01-14')
    3rd element = ('2017-01-15', '2017-01-21')
    4th element = ('2017-01-22', '2017-01-28')
    5th element = ('2017-02-01', '2017-02-07')
    
    Ex de uso (com flag):
    intervals_date_range_generator('2017-01-01', '2017-02-07', 7, consider_diff_month_intervals = 1 )

    1st element = ('2017-01-01', '2017-02-07')
    2nd element = ('2017-01-08', '2017-01-14')
    3rd element = ('2017-01-15', '2017-01-21')
    4th element = ('2017-01-22', '2017-01-28')
    5th element = ('2017-01-29, '2017-02-04')
    6th element = ('2017-02-05', '2017-02-07')
    
    """
    start, end, span = str_range_params_to_datetime(start, end, span_days)
    stop = end - (span- pd.Timedelta(days =1))
    intervals = []
    def check_and_update(start,prox,end):
        if start.month == prox.month:
            if prox < end:
                intervals.append([start, prox])
                start = prox + DATE_STEP
            return start, prox, end
        else:
            start = prox.replace(day=1)
            prox = start + span - pd.Timedelta(days = 1)
            if prox < end:
                return check_and_update(start,prox,end)
        return start, prox, end
    
    if not consider_diff_month_intervals:
        while start < stop:
            prox = start + span - pd.Timedelta(days = 1)
            start, prox, end = check_and_update(start,prox,end)

    else:
        while start < stop:
            prox = start + span - pd.Timedelta(days = 1)
            if prox < end:
                intervals.append([start, prox])
                start = prox + DATE_STEP

    intervals.append([start, end])
    
    return intervals

def gera_ppivot_nan_intervalos(ppivot_date_cols_no_time, start_date, end_date, span_days = 7, consider_diff_month_intervals = 0):
    '''
    Gera tabela com percentual de NAN do produto por tamanho da janela/periodo(span_days). 
    Ex: Percentual de NAN pra cada semana
    '''    
    intervals_generator = intervals_date_range_generator(start_date, end_date, span_days, consider_diff_month_intervals)
    valores = []
    cont = 0
    cols_names = []

    for i in intervals_generator:
        cont += 1
        if cont > 4:
            cont = 1
        percent_nan = round(ppivot_date_cols_no_time[pd.date_range(i[0],i[1], freq='D')].isnull().sum(axis = 1)/span_days * 100,1)
        if span_days == 7:
            col_name = '%NAN Sem_' + str(cont) + ' ' + i[0].strftime("%d/%m")  + ' - '  +  i[1].strftime("%d/%m")
        else:
            col_name = '%NAN Interv_' + str(cont) + ' ' + i[0].strftime("%d/%m")  + ' - '  +  i[1].strftime("%d/%m")
        cols_names.append(col_name)
        valores.append(percent_nan)
        price_pivot_nan = pd.DataFrame(valores).T
        price_pivot_nan.columns = cols_names
    
    return price_pivot_nan

def gera_ppivot_bool_intervalos(price_pivot_nan, percent_max_nan = 70):
    '''
    Gera uma pivot_table com o booleanos TRUE para cada semana que identificam se o % NAN está acima 
    do % maximo definido.
    '''
    # função para retornar o booleano de cada intervalo, por exemplo, semana (verifica se o %NAN está acima de 70%)
    price_pivot_nan_bool = price_pivot_nan.apply(lambda x: x > percent_max_nan, axis=1)
    
    return price_pivot_nan_bool

def get_number_intervals_over_max_nan(price_pivot_nan_bool):
    '''
    Retorna, para cada produto, o número total de TRUES (por exemplo semanas que excederam o limite percentual de NAN) no periodo
    
    (Conto o número de Trues em cada linha da price_pivot_nan_bool)
    '''
    return (price_pivot_nan_bool == True).sum(1)

def get_percent_intervals_over_max_nan(price_pivot_nan_bool):
    '''
    Retorna lista com a percentagem de "TRUES" (por produto) no período inteiro 
    (EX: número de semanas com TRUE, ou seja, produto com NAN excedendo o limite percentual, 
    dividido pelo número total de semanas do período analisado para cada produto)
    '''
    total_number_intervals = price_pivot_nan_bool.shape[1]
    number_intervals_over_max_nan = get_number_intervals_over_max_nan(price_pivot_nan_bool)
    percent_intervals_over_max_nan = number_intervals_over_max_nan/total_number_intervals * 100
    return percent_intervals_over_max_nan

def get_excluded_maintained_from_basic_filter(price_pivot, start_date, end_date, span_days = 7,consider_diff_month_intervals= 0, percent_max_nan = 70):
    '''
    Filtro de Missing descrito na documentação do IPC-W - Metodologia. O filtro que retorna os elementos que
    possuem todas as semanas com no máximo 70% de missing (mín 2 preços por semana) como os elementos mantidos
    e os elementos que não obedecem tal condição como excluidos
    Retorna também a price_pivot completa com o os booleanos (price_pivot_nan_bool)
    '''
    ppivot_date_cols_no_time = retira_primeiras_cols_converte_datetime(price_pivot)
        
    price_pivot_nan = gera_ppivot_nan_intervalos(ppivot_date_cols_no_time, start_date, end_date, span_days,consider_diff_month_intervals)
    
    price_pivot_nan_bool = gera_ppivot_bool_intervalos(price_pivot_nan, percent_max_nan)
    
    price_pivot_mantidos = price_pivot[(price_pivot_nan_bool == False).all(1)]
    
    price_pivot_excluidos = price_pivot[(price_pivot_nan_bool == True).any(1)]
    
    return price_pivot_mantidos,price_pivot_excluidos, price_pivot_nan_bool

def check_if_percent_missing_ok(price_pivot,start_date= '2017-7-1', end_date ='2017-09-27', span_days = 7,
                                percent_max_nan = 70, consider_diff_month_intervals =0 ):
    '''Çhecks if passing the price_pivot again through the basic Missing Filter it will not change'''
    price_pivot_mantidos = get_excluded_maintained_from_basic_filter(price_pivot, start_date,end_date, span_days = 7,
                                                                     consider_diff_month_intervals= 0,\
                                                                     percent_max_nan = 0) [0]  
    return (price_pivot.equals(price_pivot_mantidos))

def filter_by_percent_intervals_over_max_nan_index(price_pivot_excluidos_nan_bool, lim_pct_interv_over_max_nan):
    '''
    FILTRO 1 de recuperacao dos excluidos - INDICE
    
    Retorna somente os indices dos produtos cujo percentual de "Trues" na price_pivot_nan bool seja menor que um 
    limite definido, ou seja, número total de intervalos (por exemplo semanas) em que o produto excedeu o limite 
    de %NAN tolerado dividido pelo total de intervalos (por exemplo, semanas) seja menor que um lim_pct_interv_over_max_nan. 
    '''
    percent_intervals_over_max_nan = get_percent_intervals_over_max_nan(price_pivot_excluidos_nan_bool)
    new_index_list1 = percent_intervals_over_max_nan < lim_pct_interv_over_max_nan
    return new_index_list1

def filter_by_percent_intervals_over_max_nan(price_pivot_excluidos_nan_bool, lim_pct_interv_over_max_nan):
    '''
    FILTRO 1 de recuperacao dos excluidos
    
    Retorna somente os produtos cujo percentual de "Trues" na price_pivot_nan bool seja menor que um limite definido,
    ou seja, número total de intervalos (por exemplo semanas) em que o produto excedeu o limite de %NAN tolerado 
    dividido pelo total de intervalos (por exemplo, semanas) seja menor que um lim_pct_interv_over_max_nan. 
    '''
    index1 = filter_by_percent_intervals_over_max_nan_index(price_pivot_excluidos_nan_bool, lim_pct_interv_over_max_nan)
    price_pivot_excluidos_nan_bool_f1 = price_pivot_excluidos_nan_bool[index1]

    return price_pivot_excluidos_nan_bool_f1

def filter_consec_over_max_nan_intervals_index(price_pivot_excluidos_nan_bool):
    '''
    FILTRO 2 de recuperacao dos excluidos - INDICE
    
    Filtra a tabela para retornar somente os elementos que NÃO possuam "TRUE"s seguidos (intervalos acima do máximo
    permitido de %NAN seguidos) 
    '''
    consec_intervals_over_max_nan_pivot = price_pivot_excluidos_nan_bool & price_pivot_excluidos_nan_bool.shift(periods = 1, axis = 1)
    consec_intervals_over_max_nan_pivot.iloc[:,0].fillna(False, inplace = True) #first column would be all NAN
    new_index_list2 = (consec_intervals_over_max_nan_pivot == False).all(1)
    return new_index_list2

def filter_consec_over_max_nan_intervals(price_pivot_excluidos_nan_bool):
    '''
    FILTRO 2 de recuperacao dos excluidos
    
    Filtra a tabela para retornar somente os indices do elementos que NÃO possuam "TRUE"s seguidos 
    (intervalos acima do máximo permitido de %NAN seguidos) 
    '''
    new_index_list = filter_consec_over_max_nan_intervals_index(price_pivot_excluidos_nan_bool)
    price_pivot_excluidos_nan_bool_f2 = price_pivot_excluidos_nan_bool[new_index_list]
    return price_pivot_excluidos_nan_bool_f2

def filter_excluidos_nan_bool(price_pivot_excluidos_nan_bool,lim_pct_interv_over_max_nan,filt_sem_consec = True):
    '''
    FILTRO 1 + FILTRO 2 de recuperacao de excluidos
    Aplica o F1 e F2 (filter_by_percent_intervals_over_max_nan e filter_consec_over_max_nan_intervals para 
    retornar os produtos excluidos que foram recuperados).
    Esse filtro recupera por default produtos que possuem no máximo n semanas (definidas pelo segundo argumento)
    com %NAN maior que o limite (F1), sendo também essas semanas não consecutivas (F2) se o terceiro argumento for
    TRUE. 
    Dessa forma esses novos produtos recuperados dos excluidos podem ser utilizados ao fazer a imputação de preços
    buscando diminuir o enviesamento adicionado por esta aos cálculos.    
    '''
    p_excluidos_f1 = filter_by_percent_intervals_over_max_nan(price_pivot_excluidos_nan_bool, lim_pct_interv_over_max_nan)
    #p_excluidos_f1 = p_f1.copy()
    if (filt_sem_consec): 
        p_excluidos_f2 = filter_consec_over_max_nan_intervals(p_excluidos_f1)
    else:
        p_excluidos_f2 = p_excluidos_f1
    return p_excluidos_f2

def replace_missing_values(ppivot_date_cols_no_time, fwd_fill = True, back_fill = True, lim_imputados = None):
    '''
    Esta função faz o input de valores nas células com missing values.
    Parâmetros:
        fwd_fill = True --> Replica (para cada linha do dataframe) o último valor anterior preenchido da mesma linha
        (ou seja, faz o input utilizando o valor à ESQUERDA do missing). DEFAULT: TRUE;
        
        back_fill = True --> Replica (para cada linha do dataframe) o primeiro próximo valor preenchido da mesma linha
        (ou seja, faz o input utilizando o valor à DIREITA do missing). DEFAULT: TRUE;
        
        lim_imputados --> Define a quantidade máxima de valores faltantes que devem ser substituídos. 
        DEFAULT: 2  
        OBS: Para não impor limite(imputar todos os NAN) coloque lim_imputados = None (juntamente com back_fill e Fwd_fi)
    '''
    ppivot_date_cols_no_time_with_replace = ppivot_date_cols_no_time.copy()
    if fwd_fill == True:
        ppivot_date_cols_no_time_with_replace = ppivot_date_cols_no_time.ffill(axis=1, limit=lim_imputados)
        if back_fill == True:
            ppivot_date_cols_no_time_with_replace = ppivot_date_cols_no_time_with_replace.bfill(axis=1, limit=lim_imputados)        
    else:
        if back_fill == True:
            ppivot_date_cols_no_time_with_replace = ppivot_date_cols_no_time.bfill(axis=1, limit=lim_imputados)
    return ppivot_date_cols_no_time_with_replace


def filtra_missing_intervalo(price_pivot, start_date, end_date, span_days = 7,consider_diff_month_intervals = 0, percent_max_nan = 70,
    impute_val = False, lim_pct_interv_over_max_nan = 20, filt_sem_consec = True, fwd_fill = True, back_fill = True, lim_imputados = None):   
    price_pivot_mantidos, price_pivot_excluidos, price_pivot_nan_bool = get_excluded_maintained_from_basic_filter( price_pivot,
                                                                                                                   start_date, 
                                                                                                                   end_date,
                                                                                                                   span_days, 
                                                                                                                   consider_diff_month_intervals, 
                                                                                                                   percent_max_nan)    
    if (impute_val == True):
        
        price_pivot_excluidos_nan_bool = price_pivot_nan_bool[(price_pivot_nan_bool == True).any(1)]
        
        price_pivot_excluidos_filtered_bool = filter_excluidos_nan_bool(price_pivot_excluidos_nan_bool,lim_pct_interv_over_max_nan, filt_sem_consec = True)
        
        price_pivot_excluidos_recuperados = price_pivot.loc[price_pivot_excluidos_filtered_bool.index]
        
        price_pivot_excluidos_new_index = list(set(price_pivot_excluidos.index.values) - set(price_pivot_excluidos_recuperados.index.values))
        
        price_pivot_excluidos = price_pivot.loc[price_pivot_excluidos_new_index]
        
        price_pivot_mantidos_new = price_pivot.loc[(price_pivot_mantidos.index).append(price_pivot_excluidos_recuperados.index)]
        
        price_pivot_mantidos_new = replace_missing_values(price_pivot_mantidos_new, fwd_fill,back_fill, lim_imputados)
        price_pivot_mantidos = price_pivot_mantidos_new
        
        # Após imputação verifica-se novamente se os percent(70%) max missing por semana passa     
        
        is_ok = check_if_percent_missing_ok(price_pivot_mantidos,start_date, end_date, span_days, percent_max_nan,consider_diff_month_intervals)
        if (not is_ok):
            print("Tabela com imputação não passa no filtro de percent(70%) de missing em todas as semanas")
        
    return price_pivot_mantidos, price_pivot_excluidos

def filtro_missing_semanal(price_pivot, dia_inicial, dia_final , consider_diff_month_intervals = 0,
                          percent_max_nan= 70, impute_val = False, lim_pct_interv_over_max_nan = 20, filt_sem_consec = True,
                          fwd_fill = True, back_fill = True, lim_imputados = 2):
    '''
    span_days = 7
    '''
    price_pivot_mantidos, price_pivot_excluidos = filtra_missing_intervalo(price_pivot, dia_inicial, dia_final, 
                                                                              7,consider_diff_month_intervals,
                                                                              percent_max_nan,impute_val,
                                                                              lim_pct_interv_over_max_nan,
                                                                              filt_sem_consec,
                                                                              fwd_fill,back_fill,lim_imputados)
    
    return price_pivot_mantidos, price_pivot_excluidos

def filtro_missing_mensal(price_pivot, dia_inicial, dia_final, span_days = 28, consider_diff_month_intervals = 0,
                          percent_max_nan= 70, impute_val = False, lim_pct_interv_over_max_nan = 20, filt_sem_consec = True,
                          fwd_fill = True, back_fill = True, lim_imputados = 2):
    '''
    Entrada:
    price_pivot: pd dataframe 
    start_date: data formato Y%m%d
    end_date: data formato Y%m%d 
    span_days = 28 : dias do periodo (como é mês default é 28 dias)
    consider_diff_month_intervals = 0 : (se considero intervalos com meses diferentes entre a dia inicial e final)
    percent_max_nan= 70 : percentual máximo de NAN suportavel por intervalo (mês)
    impute_val = False: 
    lim_pct_interv_over_max_nan = 20
    filt_sem_consec = True
    fwd_fill = True 
    back_fill = True
    lim_imputados = 2
    
    Saída: price_pivot_mantidos, price_pivot_excluidos
    '''
    price_pivot_mantidos, price_pivot_excluidos = filtra_missing_intervalo(price_pivot, dia_inicial, dia_final, 
                                                                              span_days,consider_diff_month_intervals,
                                                                              percent_max_nan,impute_val,
                                                                              lim_pct_interv_over_max_nan,
                                                                              filt_sem_consec,
                                                                              fwd_fill,back_fill,lim_imputados)
    
    return price_pivot_mantidos, price_pivot_excluidos  


# In[ ]:



