import export_to_pivot as exp
import numpy as np
import pandas as pd

DATE_FORMAT = '%Y-%m-%d' # See http://strftime.org/
DATE_STEP = pd.Timedelta(days=1)

price_export_csv_file = "price_export_v4.csv"

price_pivot = exp.export_to_pivot(price_export_csv_file, drop_regular_prices = False, intraday_aggfunc = 'mean')
datas_faltantes = exp.dias_faltantes(price_pivot)

#função que calcule apenas as médias "gera_medias", e depois recriar a "gera_relativos" usando essa "gera_medias".

def gera_medias(ppivot, mensal=0, span_days=1):
    '''
    Retorna, para cada produto, a média de preços de cada dia (de acordo com o período de referência definidos pelo parâmetro span_days).
    Caso mensal = 1, retorna as médias de preços mensais.

    Argumentos:
    mensal (0 ou 1) -- Indica se o cálculo das médias será o consolidado mensal.
    ppivot -- Tabela contendo os produtos e os preços diários que serão base para o cálculo das médias.
    span_days -- Quantidade de dias a "varrer" para trás (incluindo o próprio dia), de forma a calcular a média de preços.
    '''
    infos = ppivot.iloc[:, :6]
    prices = ppivot.iloc[:, 6:]

    if mensal == 1:
        ppivot_2 = prices.T
        ppivot_2.index = pd.to_datetime(ppivot_2.index)
        ppivot_2 = ppivot_2.astype(float)
        month_mean = ppivot_2.resample('M').mean().T
        month_mean.columns = pd.to_datetime(month_mean.columns).strftime('%Y-%b')
        full_month_mean = pd.merge(infos, month_mean, right_index=True, left_index=True)
        ppivot = full_month_mean.copy()
        prices = ppivot.iloc[:, 6:]
        span_days = 1

    ppivot = ppivot.copy()  # para evitar o SettingWithCopyWarning
    ppivot_medias = prices.rolling(window=span_days, min_periods=0, center=False, axis=1).mean()
    ppivot.loc[:, 6:] = ppivot_medias
    return ppivot


def gera_relativos(ppivot, mensal=0, span_days=1, back_steps=1):
    '''
    Retorna, para cada produto, os relativos de cada dia (de acordo com os períodos de referência definidos pelos parâmetros span_days e back_steps).
    Caso mensal = 1, retorna os relativos mensais.

    Argumentos:
    mensal (0 ou 1) -- Indica se o cálculo dos relativos será o consolidado mensal.
    ppivot -- Tabela contendo os produtos e os preços diários que serão base para o cálculo dos relativos.
    span_days -- Quantidade de dias a "varrer" para trás (incluindo o próprio dia), de forma a calcular a média de preços.
    back_steps -- Quantidade de células a pular para trás, de forma a calcular o relativo.
    '''
    ppivot_medias = gera_medias(ppivot, mensal, span_days)

    infos = ppivot_medias.iloc[:, :6]
    means = ppivot_medias.iloc[:, 6:]

    if mensal == 1:
        back_steps = 1

    ppivot_relativos = means.pct_change(periods=back_steps,
                                        axis=1) + 1  # esse "+1" é necessário pois a função pct_change retorna a variação percentual
    ppivot_medias.iloc[:, 6:] = ppivot_relativos
    return ppivot_medias


#Função para calcular a média geométrica de uma lista, sem considerar os NANs

np.seterr(all='ignore')

def nangmean(arr):
    with np.errstate(divide='ignore'):
        arr = np.asarray(arr)
        inverse = 1 / np.sum(~np.isnan(arr))
        apoio = inverse * np.nansum(np.log(arr))
        nangmean = np.exp(apoio)
        return nangmean

# cálculo da inflação mensal (média geométrica dos relativos)

#cálculo da inflação com base em dias corridos (média geométrica dos relativos de dias corridos):
def gera_inflacao_corridos(ppivot, subitem=None, day=None, span_days=5, back_steps=1):
    '''
    Retorna, para cada subitem, a inflacao com base em dias corridos.

    Argumentos:
    ppivot -- Tabela contendo os produtos e os preços diários que serão base para o cálculo da inflacao.
    subitem (OPCIONAL) -- Filtra o subitem a ser consultado. Ex.: 'acem', 'pao_forma', 'arroz'.
    day (OPCIONAL) -- Formato:(aaaa-mm-dd) --- retorna a inflação do dia consultado.
    span_days -- Quantidade de dias a "varrer" para trás (incluindo o próprio dia), de forma a calcular a média de preços.
    back_steps -- Quantidade de células a pular para trás, de forma a calcular o relativo.

    obs.: caso 'subitem' e 'day' sejam None, a função retorna as inflações de todos os dias e todos os subitens.

    Returns
    -------
    table : DataFrame
    '''
    relativos = gera_relativos(ppivot, mensal=0, span_days=span_days, back_steps=back_steps)
    data1 = relativos.loc[:, ['ipc_subitem_name']]  # pegando apenas a coluna "ipc_subitem_name"
    data2 = relativos.iloc[:, 6:]  # parte do dataframe que contém apenas os relativos
    data = pd.merge(data1, data2, right_index=True, left_index=True)

    if subitem is None:
        if day is None:
            inflacao = pd.pivot_table(data, index=['ipc_subitem_name'], aggfunc=nangmean)
        else:
            inflacao = data.groupby('ipc_subitem_name')[day].apply(nangmean)
    else:
        if day is None:
            inflacao = pd.pivot_table(data, index=['ipc_subitem_name'], aggfunc=nangmean)
            inflacao = inflacao.reset_index()
            inflacao = inflacao[inflacao.ipc_subitem_name == subitem]
            inflacao = inflacao.set_index('ipc_subitem_name')
        else:
            inflacao = data[data.ipc_subitem_name == subitem].groupby(group_by)[day].apply(nangmean)

    inflacao = inflacao.reset_index()
    return pd.DataFrame(inflacao)

def main():
    print(gera_inflacao_corridos(price_pivot, subitem=None, day=None, span_days=5, back_steps=1))

if __name__ == '__main__':
    main()