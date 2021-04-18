# bigQuant 跳空高开追涨策略,深度研究风控方法


start_test = '2018-02-01'
end_test = '2019-05-01'
start_trade = '2019-10-23'
end_trade = '2020-04-01'

log = T.BigLogger('test')


# Python 代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端
def m6_run_bigquant_run(input_1, input_2, input_3):
    # 示例代码如下。在这里编写您的代码
    # df = pd.DataFrame({'data': [1, 2, 3]})
    # data_1 = DataSource.write_df(df)
    # data_2 = DataSource.write_pickle(df)
    # return Outputs(data_1=data_1, data_2=data_2, data_3=None)
    df = input_1.read_df()
    ins = m1.data.read_pickle()['instruments']
    start = m1.data.read_pickle()['start_date']
    end = m1.data.read_pickle()['end_date']

    df1 = D.features(ins, start, end, fields=['close_0', 'high_1', 'open_0', 'low_0', "st_status_0"])
    df_final = pd.merge(df, df1, on=['date', 'instrument'])
    df_final = df_final[df_final['st_status_0'] == 0]
    df_final = df_final[df_final['low_0'] > df_final['high_1']]
    df_final = df_final[df_final['close_0'] > df_final['open_0']]
    log.info('用于训练的样本的总个数:' + str(len(df_final)))
    data_1 = DataSource.write_df(df_final)
    return Outputs(data_1=data_1, data_2=None, data_3=None) \
 \
        # Python 回测代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端


def m11_run_bigquant_run(input_1, input_2, input_3):
    # 示例代码如下。在这里编写您的代码
    # df = pd.DataFrame({'data': [1, 2, 3]})
    # data_1 = DataSource.write_df(df)
    # data_2 = DataSource.write_pickle(df)
    # return Outputs(data_1=data_1, data_2=data_2, data_3=None)
    df = input_1.read_df()
    ins = m9.data.read_pickle()['instruments']
    start = m9.data.read_pickle()['start_date']
    end = m9.data.read_pickle()['end_date']

    df1 = D.features(ins, start, end, fields=['close_0', 'high_1', 'open_0', 'low_0', "st_status_0"])
    df_final = pd.merge(df, df1, on=['date', 'instrument'])
    df_final = df_final[df_final['st_status_0'] == 0]
    df_final = df_final[df_final['low_0'] > df_final['high_1'] + 0.02]
    #     df_final=df_final[df_final['low_0']>df_final['high_1']]
    df_final = df_final[df_final['close_0'] > df_final['open_0']]

    ('用于训练的样本的总个数:', len(df_final))
    data_1 = DataSource.write_df(df_final)
    return Outputs(data_1=data_1, data_2=None, data_3=None)


# 后处理函数，可选。输入是主函数的输出，可以在这里对数据做处理，或者返回更友好的outputs数据格式。此函数输出不会被缓存。
def m6_post_run_bigquant_run(outputs):
    return outputs


# 后处理函数，可选。输入是主函数的输出，可以在这里对数据做处理，或者返回更友好的outputs数据格式。此函数输出不会被缓存。
def m11_post_run_bigquant_run(outputs):
    return outputs


# 回测引擎：初始化函数，只执行一次
def m19_initialize_bigquant_run(context):
    # 加载预测数据
    context.ranker_prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))
    # 预测数据，通过options传入进来，使用 read_df 函数，加载到内存 (DataFrame)
    # 设置买入的股票数量，这里买入预测股票列表排名靠前的5只
    stock_count = 2
    # 每只的股票的权重，如下的权重分配会使得靠前的股票分配多一点的资金，[0.339160, 0.213986, 0.169580, ..]
    #     context.stock_weights=(1,0)
    context.stock_weights = T.norm([1 / math.log(i + 2) for i in range(0, stock_count)])
    # 设置每只股票占用的最大资金比例
    context.max_cash_per_instrument = 0.5
    context.options['hold_days'] = 1
    # 用于判断奇偶交易
    context.datecont = 0


# 回测引擎：每日数据处理函数，每天执行一次
def m19_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    ranker_prediction = context.ranker_prediction[
        context.ranker_prediction.date == data.current_dt.strftime('%Y-%m-%d')]

    if context.datecont == 0:
        context.datecont = 1
    else:
        context.datecont = 0
    # 大盘风控模块,读取风控数据
    today = data.current_dt.strftime('%Y-%m-%d')
    benckmark_ret0 = context.benckmark_ret0[today]
    benckmark_ret1 = context.benckmark_ret1[today]
    benckmark_ret2 = context.benckmark_ret2[today]
    benckmark_ret3 = context.benckmark_ret3[today]
    benckmark_risk_v0 = context.benckmark_risk_v0[today]
    benckmark_risk_v1 = context.benckmark_risk_v1[today]
    benckmark_risk_v2 = context.benckmark_risk_v2[today]
    risk = 0
    if benckmark_ret0 < 0.001:
        if benckmark_risk_v0 > 0:
            log.info(str(today) + '大盘放量下跌,全仓卖出')
            risk = 1
        elif benckmark_ret1 < 0.001 and benckmark_ret2 < 0.002:
            log.info(str(today) + '大盘放量下跌,全仓卖出')
            risk = 1
        if benckmark_ret3 < -0.02:
            log.info(str(today) + '大盘3日下跌超过2%,全仓卖出')
            risk = 1
    if benckmark_ret0 < 0.001:
        if (benckmark_risk_v0 + benckmark_risk_v1) < 0:
            log.info(str(today) + '大盘缩量上涨,全仓卖出')
            risk = 1

    if risk == 1:
        positions_all = [equity.symbol for equity in context.portfolio.positions]
        if len(positions_all) > 0:
            # 全部卖出后返回
            for i in positions_all:
                if data.can_trade(context.symbol(i)):
                    context.order_target_percent(context.symbol(i), 0)
    # 大盘风控结束

    # 1. 资金分配
    # 平均持仓时间是hold_days，每日都将买入股票，每日预期使用 1/hold_days 的资金
    # 实际操作中，会存在一定的买入误差，所以在前hold_days天，等量使用资金；之后，尽量使用剩余资金（这里设置最多用等量的1.5倍）
    is_staging = context.trading_day_index < context.options['hold_days']  # 是否在建仓期间（前 hold_days 天）
    cash_avg = context.portfolio.portfolio_value / context.options['hold_days']
    cash_for_buy = min(context.portfolio.cash, (1 if is_staging else 1.5) * cash_avg)
    cash_for_sell = cash_avg - (context.portfolio.cash - cash_for_buy)
    positions = {e.symbol: p.amount * p.last_sale_price
                 for e, p in context.portfolio.positions.items()}

    # 2. 生成卖出订单：hold_days天之后才开始卖出；对持仓的股票，按机器学习算法预测的排序末位淘汰
    # if not is_staging and cash_for_sell > 0:
    equities = {e.symbol: p for e, p in context.portfolio.positions.items() if p.amounts > 0}
    if (len(equities) > 0):
        for i in equities.keys():
            last_sale_date = equities[i].last_sale_date
            delta_days = data.current_dt - last_sale_date
            hold_days = delta_days.days
            if hold_days > 0:
                context.order_target(context.symbol(i), 0)

    # 3. 生成买入订单：按机器学习算法预测的排序，买入前面的stock_count只股票
    buy_cash_weights = context.stock_weights
    buy_instruments = list(ranker_prediction.instrument[:len(buy_cash_weights)])
    max_cash_per_instrument = context.portfolio.portfolio_value * context.max_cash_per_instrument
    for i, instrument in enumerate(buy_instruments):
        try:
            cash = cash_for_buy * buy_cash_weights[i]
            if cash > max_cash_per_instrument - positions.get(instrument, 0):
                # 确保股票持仓量不会超过每次股票最大的占用资金量
                cash = max_cash_per_instrument - positions.get(instrument, 0)
            if context.datecont == 1:
                #             if context.datecont >= 0:
                #                 获取今天和昨天的成交量
                volume_since_buy = data.history(context.symbol(instrument), 'volume', 3, 'id')
                close_price = data.current(context.symbol(instrument), 'close')  # 当前收盘价
                #                 print('open_price=',open_price)
                high_price = data.current(context.symbol(instrument), 'high')
                #                 print('high_price=',high_price)

                if ((volume_since_buy[2] / volume_since_buy[1] < 2.5) or (high_price / close_price < 1.05)) and \
                        volume_since_buy[2] / volume_since_buy[0] > 1:
                    current_price = data.current(context.symbol(instrument), 'price')
                    amount = math.floor(cash / current_price - cash / current_price % 100)
                    context.order(context.symbol(insturment), amount)
                    return
                else:
                    print('today=', today, 'instrument=', instrument)
        except:
            print('today=', today, 'instrument=', instrument)


# 回测引擎：准备数据，只执行一次
def m19_prepare_bigquant_run(context):
    start_date = (pd.to_datetime(context.start_date) - datetime.timedelta(days=10)).strftime('%Y-%m-%d')
    # 多取10天的数据便于后面计算均值
    benckmark_data = D.history_data(instruments=['000001.SHA'], start_date=start_test, end_date=context.end_date)
    # 计算指数0日,3日涨幅
    benckmark_data['ret0'] = benckmark_data['close'] / benckmark_data["close"].shift(1) - 1
    benckmark_data['ret3'] = benckmark_data['close'] / benckmark_data["close"].shift(3) - 1
    benckmark_data['risk_v0'] = benckmark_data['volume'] / benckmark_data["volume"].shift(1) - 1

    benckmark_data['date'] = benckmark_data['date'].apply(lambda x: x.strftime('%Y-%m-%d'))
    # 设置日期为索引
    benckmark_data.set_index('date', inplace=True)
    # 把风控序列输出给全局变量context.benckmark_risk
    context.benckmark_ret0 = benckmark_data['ret0']
    context.benckmark_ret1 = benckmark_data['ret0'].shift(1)
    context.benckmark_ret2 = benckmark_data['ret0'].shift(2)
    context.benckmark_ret3 = benckmark_data['ret3']
    context.benckmark_risk_v0 = benckmark_data['risk_v0']
    context.benckmark_risk_v1 = benckmark_data['risk_v0'].shift(1)
    context.benckmark_risk_v2 = benckmark_data['risk_v0'].shift(2)


m1 = M.instruments.v2(
    start_date=start_test,
    end_date=end_test,
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m2 = M.advanced_auto_labeler.v2(
    instruments=m1.data,
    label_expr="""# #号开始的表示注释
# 0. 每行一个，顺序执行，从第二个开始，可以使用label字段
# 1. 可用数据字段见 https://bigquant.com/docs/develop/datasource/deprecated/history_data.html
#   添加benchmark_前缀，可使用对应的benchmark数据
# 2. 可用操作符和函数见 `表达式引擎 <https://bigquant.com/docs/develop/bigexpr/usage.html>`_

# 计算收益：5日收盘价(作为卖出价格)除以明日开盘价(作为买入价格)
shift(close, -5) / shift(open, -1)

# 极值处理：用1%和99%分位的值做clip
clip(label, all_quantile(label, 0.01), all_quantile(label, 0.99))

# 将分数映射到分类，这里使用20个分类
all_wbins(label, 20)

# 过滤掉一字涨停的情况 (设置label为NaN，在后续处理和训练中会忽略NaN的label)
where(shift(high, -1) == shift(low, -1), NaN, label)
""",
    start_date='',
    end_date='',
    benchmark='000300.SHA',
    drop_na_label=True,
    cast_label_int=True
)

m3 = M.input_features.v1(
    features="""
# #号开始的表示注释
# 多个特征，每行一个，可以包含基础特征和衍生特征
return_5
return_10
return_20
avg_amount_0/avg_amount_5
avg_amount_5/avg_amount_20
rank_avg_amount_0/rank_avg_amount_5
rank_avg_amount_5/rank_avg_amount_10
rank_return_0
rank_return_5
rank_return_10
rank_return_0/rank_return_5
rank_return_5/rank_return_10
pe_ttm_0
"""
)

m15 = M.general_feature_extractor.v7(
    instruments=m1.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=90
)

m16 = M.derived_feature_extractor.v3(
    input_data=m15.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False
)

m7 = M.join.v3(
    data1=m2.data,
    data2=m16.data,
    on='date,instrument',
    how='inner',
    sort=False
)

m5 = M.dropnan.v2(
    input_data=m7.data
)

m6 = M.cached.v3(
    input_1=m5.data,
    run=m6_run_bigquant_run,
    post_run=m6_post_run_bigquant_run,
    input_ports='',
    params='',
    output_ports=''
)

m4 = M.stock_ranker_train.v6(
    training_ds=m6.data_1,
    features=m3.data,
    learning_algorithm='排序',
    number_of_leaves=30,
    minimum_docs_per_leaf=1000,
    number_of_trees=20,
    learning_rate=0.1,
    max_bins=1023,
    feature_fraction=1,
    data_row_fraction=1,
    ndcg_discount_base=1,
    m_lazy_run=False
)

m9 = M.instruments.v2(
    start_date=T.live_run_param('trading_date', start_trade),
    end_date=T.live_run_param('trading_date', end_trade),
    market='CN_STOCK_A',
    instrument_list='',
    max_count=0
)

m17 = M.general_feature_extractor.v7(
    instruments=m9.data,
    features=m3.data,
    start_date='',
    end_date='',
    before_start_days=90
)

m18 = M.derived_feature_extractor.v3(
    input_data=m17.data,
    features=m3.data,
    date_col='date',
    instrument_col='instrument',
    drop_na=False,
    remove_extra_columns=False
)

m11 = M.cached.v3(
    input_1=m18.data,
    run=m11_run_bigquant_run,
    post_run=m11_post_run_bigquant_run,
    input_ports='',
    params='{}',
    output_ports=''
)

m10 = M.dropnan.v2(
    input_data=m11.data_1
)

m8 = M.stock_ranker_predict.v5(
    model=m4.model,
    data=m10.data,
    m_lazy_run=False
)

m19 = M.trade.v4(
    instruments=m9.data,
    options_data=m8.predictions,
    start_date='',
    end_date='',
    initialize=m19_initialize_bigquant_run,
    handle_data=m19_handle_data_bigquant_run,
    prepare=m19_prepare_bigquant_run,
    volume_limit=0.025,
    order_price_field_buy='open',
    order_price_field_sell='close',
    capital_base=1000000,
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    product_type='股票',
    plot_charts=True,
    backtest_only=False,
    benchmark='000300.SHA'
)