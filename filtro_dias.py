
# coding: utf-8

# In[57]:

from export_to_pivot import dias_faltantes
import export_to_pivot as exp
import pandas as pd

def filtro_dias(price_pivot, dia_inicial='01-Jun-17', dia_final = 'today', lista_dias_remov = ["2017-07-15", "2017-08-22"], periodo_dias_remov = []):
    """
    1.Reduz a price_pivot para o periodo de dias determinado
	2.Remove periodo de dias (per) e lista de dias pontuais da price_table
    3.Completa os dias que foram eliminados com dias com NAN
    Entradas:
        price_pivot: pivot_table
        dia_inicial: Dia inicial a partir do qual começará a price_pivot filtrada (inicia-se em Junho por padrao)
        dia_final: Último dia a se contar para a price_pivot filtrada (default = most_updated)
        lista_dias_remov: Lista de dias pontuais para remoção (dias de quedas de servidor por exemplo)
        periodo_dias_remov: Lista no formato [dia_inicial, dia_final] indicando o periodo para remoção de valores
        
    Saída:
        price_pivot: Uma tabela histórico de preços na forma pivot_table com os dias filtrados.
    """

    #Remove algum periodo de dias (periodos_remov)
    #Remove os dias pontuais de queda de servidor (lista_dias_remov)
    #Removing days from lista_dias_remov
    remov_lista_dias(price_pivot, lista_dias_remov)
    #Removing days from periodo_dias_remov 
    remov_periodo_dias(price_pivot, periodo_dias_remov)
    #Filtering period
    price_pivot = filtro_periodo_dias(price_pivot, dia_inicial,dia_final)
    
    #Verificar os dias que foram retirados e recoloca-los completá-los na pivot_table com NAN
    datas_faltantes = dias_faltantes(price_pivot)
    price_pivot = exp.fill_datas_faltantes(price_pivot, datas_faltantes)
    
    return price_pivot
    

def remov_lista_dias(price_pivot, lista_dias_remov = ["2017-07-15", "2017-08-22"]):
    """
    Funcao que remove uma lista de dias inplace. Obs: Colocar apenas dias presentes na pivot table.
   
   Entradas:
       price_pivot: Histórico de preços no formato pivot_table
       
       lista_dias_remov: lista com elementos na forma de string "Ano-mês-dia", esses dias serão removidos
       
    Saída:
        Nenhuma. A mudança é feita inplace.
    """
    lista_dias_remov_tratada = []2
    for dia in lista_dias_remov:
        lista_dias_remov_tratada.append(pd.to_datetime(dia).strftime('%Y-%m-%d'))
    price_pivot.drop([col for col in price_pivot.columns if col in lista_dias_remov_tratada], axis=1, inplace=True)


def remov_periodo_dias(price_pivot, periodo_dias_remov = []): 
    """
    Funcao que remove um periodo de dias inplace do price_pivot. Obs: Colocar apenas dias presentes na pivot table.
    
    Entradas:
       price_pivot: Histórico de preços no formato pivot_table
       
       periodo_dias_remov: lista com dois elementos em formato de string "Ano-mês-dia", o dia inicial e o dia final 
       do periodo que deseja-se remover
       
    Saída:
        Nenhuma. A mudança é feita inplace.
    """
    if(len(periodo_dias_remov)==2): 
        idx_periodo_remov_inicio = price_pivot.columns.get_loc(pd.to_datetime(periodo_dias_remov[0]).strftime('%Y-%m-%d'))
        idx_periodo_remov_fim = price_pivot.columns.get_loc(pd.to_datetime(periodo_dias_remov[1]).strftime('%Y-%m-%d'))

        price_pivot.drop(price_pivot.columns[idx_periodo_remov_inicio:idx_periodo_remov_fim + 1], axis = 1, inplace = True)
    elif(len(periodo_dias_remov) ==0):
        pass
    else:
        print("Coloque apenas dois elementos na lista de periodo_dias_remov")


def filtro_periodo_dias(price_pivot, dia_inicial='1-Jun-2017', dia_final='today'):
    """
    Filtra um periodo de dias apenas.
    
    Entradas:
        price_pivot: histórico de preços em formato pivot_table
        dia_inicial: string com dia inicial no formato "Ano-mes-dia". Ex: 1-Jun-2017
        dia_final: string com a dia final no mesmo formato. O valor default é pegar até o último dia da pivot_table.
        
    Saída: Nova pivot_table com a filtragem (apenas dias dentro do periodo dado)
    """    
    
    dia_inicial = pd.to_datetime(dia_inicial).strftime('%Y-%m-%d')

    if(dia_inicial in dias_faltantes(price_pivot)): #Se o dia inicial entrado não está em alguma coluna da pivot_table
        print("Coloque um dia_inicial que esteja presente na pivot_table")

    if dia_final == 'today':
        dia_final = price_pivot.columns[-1] #pega até o final da coleta
    else:
        dia_final = pd.to_datetime(dia_final).strftime('%Y-%m-%d')
        if(dia_final in dias_faltantes(price_pivot)):
            print("Coloque um dia_inicial ou dia_final que esteja presente na pivot_table")

    data_columns = price_pivot.loc[:,'website_id':'ipc_subitem_id']
    filtered_days_columns = price_pivot.loc[:, dia_inicial:dia_final]

    price_pivot = pd.concat([data_columns,filtered_days_columns], axis=1)
    
    return price_pivot