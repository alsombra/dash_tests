import export_to_pivot as exp
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

def main():
    print(gera_relativos(price_pivot, mensal=0, span_days=1, back_steps=1))

if __name__ == '__main__':
    main()