# Author: Renato Santos Aranha
# Projeto IPC-W FGV
# Last version: 03/10/2018

import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
#import dash_auth
import plotly.graph_objs as go
import numpy as np
import pandas as pd
import dash_table_experiments as dt
from nltk.tokenize import word_tokenize
import gera_inflacao_mes as gim
import nltk

nltk.download('punkt')


login_list = [
    ['ipcw_fgv_', '321321']
]

# app = dash.Dash('auth')
app = dash.Dash(__name__)

'''
auth = dash_auth.BasicAuth(
    app,
    login_list
)
'''

config = app.config['suppress_callback_exceptions'] = True
server = app.server

# Read data for tables (one df per table)

df = round(gim.gera_inflacao_mes(gim.price_pivot, subitem=None, ano_mes=None), 5)
ppivot = gim.price_pivot  # CARREGAR A base correta  - "base-pre-classificacao"

lista_subitem_name = [
    {'label': value, 'value': value} for key, value in
    enumerate(np.unique([i for i in gim.price_pivot.ipc_subitem_name]))
]


# REUSABLE COMPONENTS

tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'font-size': 20,
    'fontWeight': 'bold'

}

whitelist_style = {
    'border': '1px solid black',
    'font-size':'10',
    'padding': '10px',
    'width': '540px'}

blacklist_style = {
    'border': '1px solid black',
    'font-size':'10',
    'width': '540px',
    'padding': '10px',
    'height': '400px',
    'overflow':'scroll'
}

products_style ={
    'border': '1px solid black',
    'font-size':'10',
    'padding': '10px',
    'width': '100%',
    'height': '100px',
    'position': 'center',
    'margin':'100 auto'

}

button_style = {
    'color': 'red'
}

list_of_itens = [html.P(["nome "+ str(i) + ' na whitelist']) for i in range(10)]


def filter_subitem(subitens):
    df1 = df[df.ipc_subitem_name.isin(subitens)]
    return df1


def print_pdf_button():
    printButton = html.A(['Print PDF']
                         , className="button no-print print"
                         , style={'position': "absolute", 'right': '117'}
                         )
    return printButton


# includes page/full view
def get_logo():
    logo = html.Div([

        html.Div([
            html.A(
                html.Img(src='http://bibliotecadigital.fgv.br/dspace/bitstream/id/45381/?sequence=-1',
                         style={'width': '130px'}),
                href='https://emap.fgv.br/',
                target="_blank"),
            print_pdf_button()
        ], className="ten columns padded")

    ], className="row gs-header")
    return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H5(
                'IPCW - Painel do projeto')
        ], className="twelve columns padded")

    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([

        dcc.Link('Overview   ', href='/overview', className="tab first"),

        dcc.Link('Price Performance   ', href='/price-performance', className="tab"),

        dcc.Link('Classificação   ', href='/classificacao', className="tab")
    ], className="row ", style={'font-size': 12, 'padding': 10})
    return menu

pricePerformance = html.Div([
    html.Div([

        # Header
        get_logo(),
        get_header(),
        html.Br([]),
        get_menu(),

        # Selector

        html.Div([
            dcc.RadioItems(
                id='options_FULL_rd',
                options=[
                    {'label': 'Indices por subitem', 'value': 'FULL'},
                    {'label': 'Raw Data', 'value': 'RD'}
                ],
                value='FULL',
                labelStyle={'width': '13%', 'display': 'inline-block'}
            )
        ]),

        html.Div(id='test_',
                 className="row"
                 ),

    ], className="row")

], className="ten columns offset-by-one")

subitem = 'acem' #Placeholder inicial

classifica = html.Div([

    html.Div([

        # Header
        get_logo(),
        get_header(),
        html.Br([]),
        get_menu(),

        # Tabs

        html.Div([
            html.Div(children=[dcc.Dropdown(
                options=lista_subitem_name,
                value='acem',
                id='dropdown-lists')],
                style={'width': '20%'})
            ,
            html.Div(id='conjuntao', className="twelve columns padded",
                     style={'padding': 10, 'width': '100%', 'margin': '0 auto'}, children=[
                    html.Div(id='filter-table', children=dt.DataTable(rows=[{}]),
                             style={'width': '450px', 'float': 'left'}),
                    html.Div(id='teste_1245',
                             style={'margin': '0 auto', 'float': 'right', 'width': '550px', 'padding': '0px'},
                             children=[dcc.Tabs(id="tabs", value='tab-w',
                                                style={'font-size': 15,
                                                       'width': '540px',
                                                       'float': 'right'},
                                                children=[
                                                    dcc.Tab(id='wl', label='Whitelist', value='tab-w',
                                                            selected_style=tab_selected_style),
                                                    dcc.Tab(id='bl', label='Blacklist', value='tab-b',
                                                            selected_style=tab_selected_style,
                                                            children=[html.Div(id='bl.container',
                                                                               className='tabs-content',
                                                                               style=blacklist_style,
                                                                               children=[
                                                                                   html.Div(
                                                                                       id='bl_container_dept_title',
                                                                                       children=html.H5(
                                                                                           ["Department: "],
                                                                                           style={
                                                                                               'width': '230px',
                                                                                               'float': 'left',
                                                                                               'font-weight': 'bold'
                                                                                           })),
                                                                                   html.Div(
                                                                                       id='bl_container_name_title',
                                                                                       children=html.H5(["Name: "],
                                                                                                        style={
                                                                                                            'width': '230px',
                                                                                                            'float': 'right',
                                                                                                            'font-weight':'bold',
                                                                                                        }
                                                                                                        )),
                                                                                   html.Div(
                                                                                       id='bl_container_department',
                                                                                       style={'width': '235px',
                                                                                              'float': 'left',
                                                                                              'overflow': 'scroll',
                                                                                              'padding': '10px'}),
                                                                                   html.Div(id='bl_container_name',
                                                                                            style={'width': '235px',
                                                                                                   'float': 'right',
                                                                                                   'overflow': 'scroll',
                                                                                                   'padding': '20px'})
                                                                                   ]
                                                                               )
                                                            ]),

                                                ]
                                                ),
                                        html.Button('Salvar Black e White lists', id = 'save_bwlists', n_clicks = 0),
                                        html.Button('Filtrar', id='filter_button',style = button_style, className = "filter_button", n_clicks = 0)
                                       ])
                ])
            ,
        html.Div(id='products_container', children = html.H4(["AQUI FICAM OS PRODUTOS!"]),style = { 'height': '100px', 'float': 'bottom'})
        ]),
    ], className="row")
], className="ten columns offset-by-one")

noPage = html.Div([

    html.Div([
        # Header
        get_logo(),
        get_header(),
        html.Br([]),
        get_menu(),
        html.Div([
            html.P(["Aba Overview."]),
            html.P(["Equipe IPCW: Ainda estamos trabalhando nesse conteúdo."])
        ], className="no-page"),
    ], className="row")
], className="ten columns offset-by-one")

# Describe the layout, or the UI, of the app
app.layout = html.Div([
    dcc.Location(id='url', pathname='', refresh=False),
    html.Div(id='page-content'),
    html.Div(dt.DataTable(rows=[{}]), style={'display': 'none'})
])


# CALLBACKS SECTION

subitem = 'sopa_desidratada' #Padrão: somente letras minusculas separados por underline (_)
subitem_id = 11111 # IBRE code
#REGEX para  case insensitive e considerando acentuação
#EXEMPLO: subitem == feijao -> "(?i)feijão|feijao"
#whitelist = {'feijão','feijao'}
#EXEMPLO2: subitem == acucar -> "(?i)açúcar|acucar|açucar|acúcar"
#whitelist = {'açúcar','acucar','açucar','acúcar'}


#white_list = {"sopa", "desidratada", " po ", " pó "}
#base_subitem = data[data['name'].str.contains("(?i)sopa|desidratada| po | pó")]

#registro do sumário do resultado da blacklist


# SHOW PRODUTOS FILTRADOS

def factorial(n):
    result = 0
    for i in range(n):
        for i in range(n):
            for i in range(n):
                for i in range(n):
                    result = result + 1
    return result

@app.callback(dash.dependencies.Output('products_container', 'children'),
              [dash.dependencies.Input('filter_button', 'n_clicks'),
              dash.dependencies.Input('dropdown-lists', 'value')])
def on_click(n_clicks, dropdown_value):
        if n_clicks > 0:
            layout = [html.H4(["Produtos Filtrados de " + dropdown_value], style={ 'text-align': 'center'}),
                      html.Div(id = "filtered_products_cotainer", style = products_style, children = factorial(66+n_clicks))]
            return layout


#OS DOIS CALLBACKS ABAIXO DA BLACKLIST E WHITELIST SÃO INTERLIGADOS



#SHOW WHITELIST (TAB CALLBACK)


@app.callback(dash.dependencies.Output('wl', 'children'),
              [dash.dependencies.Input('tabs', 'value'),
              dash.dependencies.Input('dropdown-lists', 'value')])
def func(tab_value, dropdown_value):
    subitem_whitelist_name = dropdown_value
    if tab_value == 'tab-w':
        with open('black_white_sublists/' + subitem_whitelist_name + '_whitelist.pickle', 'rb') as file:
            subitem_whitelist_loaded = pickle.load(file)
        list_whitelist = [html.P([i]) for i in subitem_whitelist_loaded]
        lista = html.Div(className='tabs-content', style=whitelist_style, children=list_whitelist)
        return lista


#SHOW BLACKLIST DEPARTMENT (TAG CALLBACK)
@app.callback(dash.dependencies.Output('bl_container_department', 'children'),
               [dash.dependencies.Input('tabs', 'value'),
                dash.dependencies.Input('dropdown-lists', 'value')])
def show_blacklist_dept(tab_value, dropdown_value):
    subitem_blacklist_name = dropdown_value
    if tab_value == 'tab-b':
        with open('black_white_sublists/' + subitem_blacklist_name + '_blacklist.pickle', 'rb') as file:
            subitem_blacklist_loaded = pickle.load(file)
        list_blacklist_depts = [html.Div([html.P([i]) for i in subitem_blacklist_loaded["department"]])]

        return list_blacklist_depts


# SHOW BLACKLIST NAME (TAG CALLBACK)
@app.callback(dash.dependencies.Output('bl_container_name', 'children'),
               [dash.dependencies.Input('tabs', 'value'),
                dash.dependencies.Input('dropdown-lists', 'value')])
def show_blacklist_name(tab_value, dropdown_value):
    subitem_blacklist_name = dropdown_value
    if tab_value == 'tab-b':
        with open('black_white_sublists/' + subitem_blacklist_name + '_blacklist.pickle', 'rb') as file:
            subitem_blacklist_loaded = pickle.load(file)
        list_blacklist_names = [html.Div([html.P([i]) for i in subitem_blacklist_loaded["name"]])]

        return list_blacklist_names


# UPDATE CLASSIFICA
@app.callback(dash.dependencies.Output('filter-table', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def update_classifica(pathname):
    if pathname == '/classificacao':
        df1 = ppivot.copy().head(100)
        df1 = df1.loc[:, 'global_product_name':'global_product_name'] # so we can use it as a dataframe of 1 column
        tokenized_names_list = [word_tokenize(df1.global_product_name[i]) for i in range(len(df1.global_product_name))]
        token_total = [token for tokenized_name in tokenized_names_list for token in tokenized_name]
        # TRATAR ESSA PARTE PRA CONTAR FREQ DIREITO!!!!
        dict_total = {i: token_total.count(i) for i in token_total}
        df = pd.DataFrame.from_dict(dict_total, orient='index').sort_values(0, axis=0, ascending=False).reset_index()
        df = df.rename(columns={df.columns[0]: 'Nome', df.columns[1]: 'Qtd'})

        table = html.Div([dt.DataTable(
            id='table_freq_id',
            rows=df.to_dict('records'),
            columns=sorted(df.columns),
            min_width=450,
            row_selectable=True,
            resizable=True,
            editable = False,
            filterable=True,
            sortable=True,
            selected_row_indices=[])
        ])
        return table

# UPDATE PAGE
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/' or pathname == '/overview':
        return noPage
    elif pathname == '/classificacao':
        return classifica
    elif pathname == '/price-performance':
        return pricePerformance
    else:
        return noPage


# RAW DATA (exibe Pivot Table)
@app.callback(dash.dependencies.Output('test_', 'children'),
              [dash.dependencies.Input('options_FULL_rd', 'value')])
def update_radio(value):
    if value == 'FULL':
        child = [html.Div([
            html.Label('Seleção múltipla de itens:'),
            dcc.Dropdown(
                id='multi_select',
                options=lista_subitem_name,
                value=[i['label'] for i in lista_subitem_name],
                multi=True),
            html.H6(["Current Indices"], className="gs-header gs-table-header padded"),
            html.Div([html.Table(id='ipcw-table')]
                     ),
            html.H6("Performance", className="gs-header gs-table-header padded"),
            dcc.Graph(
                id='graph-4',
                config={'displayModeBar': False}
            )
        ])
        ]
        return child
    elif value == 'RD':
        table = \
            html.Div([
                dcc.Input(id='text1',
                          type='search',
                          placeholder='Digite o termo desejado',
                          value=''),
                html.Div(id='table_cut')
            ])
        return table



# UPDATE TABLE (PRICE PERFORMANCE LAYOUT)
@app.callback(
    dash.dependencies.Output('ipcw-table', 'children'),
    [dash.dependencies.Input('multi_select', 'value')])
def make_dash_table(value):
    ''' Return a dash definition of an HTML table (using a Pandas dataframe) '''
    df1 = filter_subitem(value)
    df1 = df1[['ipc_subitem_name', '2017-Jun', '2017-Jul', '2017-Aug', '2017-Sep']]  # HARDCODED! **
    table = [html.Tr([html.Td(i, style={
        'backgroundColor': 'rgb(0,62,125)',
        'color': 'white',
        'font-weight': 'bold'
    }) for i in df1.columns])]
    for index, row in df1.iterrows():
        html_row = []
        for i in range(len(row)):
            html_row.append(html.Td([row[i]]))
        table.append(html.Tr(html_row))
    return table


# UPDATE GRAPH (PRICE PERFORMANCE LAYOUT)
@app.callback(
    dash.dependencies.Output('graph-4', 'figure'),
    [dash.dependencies.Input('multi_select', 'value')])
def update_figure(subitens):
    df1 = df[df.ipc_subitem_name.isin(subitens)].reset_index(drop=True)
    x_data = ['2017-Jun', '2017-Jul', '2017-Aug', '2017-Sep'] # HARDCODED!!!!
    y_data = df1['ipc_subitem_name']
    df1 = df1[['ipc_subitem_name', '2017-Jun', '2017-Jul', '2017-Aug', '2017-Sep']] # HARDCODED!!!
    traces = []
    for i in range(len(y_data)):
        traces.append(go.Scatter(
            x=x_data,
            y=df1.T[i][1:],
            name=y_data[i]
        ))
    layout = dict(hovermode='closest',
                  autosize=True,
                  font={
                      "family": "Raleway",
                      "size": 10
                  },
                  margin={
                      "r": 40,
                      "t": 40,
                      "b": 30,
                      "l": 40
                  },
                  showlegend=True,
                  titlefont={
                      "family": "Raleway",
                      "size": 10
                  },
                  xaxis={
                      "autorange": True,
                      "range": ["2007-12-31", "2017-12-31"]
                  },
                  yaxis={
                      "autorange": False,
                      "range": [0.6, 1.5],
                      "showline": True,
                      "type": "linear",
                      "zeroline": False
                  })
    return dict(data=traces, layout=layout)


# UPDATE TABLE_RD
@app.callback(dash.dependencies.Output('table_cut', 'children'),
              [dash.dependencies.Input('text1', 'value')])
def update(value):
    df1 = ppivot.copy().head(100)
    df1 = df1.iloc[:, :4].sort_values('global_product_name') # HARDCODED PRA PEGAR SÓ AS PRIMEIRAS 5 COLUNAS (0 a 4) !!
    df1 = df1[df1.global_product_name.str.contains(value, case=0)]
    table = html.Div([html.Div('{} produtos encontrados com o termo "{}"'.format(df1.shape[0], value),
                               style={'font-size': 14, 'padding': 10}),
                      dt.DataTable(
                          id='dataTable_indices',
                          rows=df1.to_dict('records'),
                          columns=sorted(df1.columns),
                          row_selectable=True,
                          resizable=True,
                          filterable=True,
                          sortable=True,
                          selected_row_indices=[])
                      ])
    return table


external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
                "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
                "https://codepen.io/bcd/pen/KQrXdb.css",
                "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
                "https://raw.githubusercontent.com/alsombra/dash_tests/master/assets/custom_style.css"
                ]

for css in external_css:
    app.css.append_css({"external_url": css})

external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
               "https://codepen.io/bcd/pen/YaXojL.js"]

for js in external_js:
    app.scripts.append_script({"external_url": js})

if __name__ == '__main__':
    app.run_server(debug=True)