#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime

import pandas as pd
from pmdarima.arima import auto_arima

flow_df = pd.read_csv('data/RawData/flow_train.csv')
flow_df = flow_df.sort_values(by=['city_code', 'district_code', 'date_dt'])

date_dt = list()
init_date = datetime.date(2018, 3, 2)
for delta in range(15):
    _date = init_date + datetime.timedelta(days=delta)
    date_dt.append(_date.strftime('%Y%m%d'))

district_code_values = flow_df['district_code'].unique()
preds_df = pd.DataFrame()
tmp_df_columns = ['date_dt', 'city_code', 'district_code', 'dwell', 'flow_in', 'flow_out']

for district_code in district_code_values:
    sub_df = flow_df[flow_df['district_code'] == district_code]
    city_code = sub_df['city_code'].iloc[0]

    predict_columns = ['dwell', 'flow_in', 'flow_out']
    tmp_df = pd.DataFrame(data=date_dt, columns=['date_dt'])
    tmp_df['city_code'] = city_code
    tmp_df['district_code'] = district_code

    for column in predict_columns:
        arima_model = auto_arima(sub_df[column], start_p=1, start_q=1, max_p=3, max_q=3, m=7,
                                 start_P=0, seasonal=True, d=1, D=1, trace=True,
                                 error_action='ignore',  # don't want to know if an order does not work
                                 suppress_warnings=True,  # don't want convergence warnings
                                 stepwise=True)

        preds = arima_model.predict(n_periods=15)
        preds = pd.Series(preds)
        tmp_df = pd.concat([tmp_df, preds], axis=1)

    tmp_df.columns = tmp_df_columns
    preds_df = pd.concat([preds_df, tmp_df], axis=0, ignore_index=True)

preds_df = preds_df.sort_values(by=['date_dt'])
preds_df.to_csv('prediction.csv', index=False, header=False)



