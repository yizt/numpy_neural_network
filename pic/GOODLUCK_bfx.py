"""
# 作者：马克维茨，审核：邢不行

# 本交易系统用于bitfinex的margin交易。本程序以usd为margin账户保证金，若以其他币种作为保证金，需要做相应的修改。

# 将本程序放在课程代码class9中，填写apikey/secret/钉钉id之后即可直接运行。

# 程序主要思路如下：

通过while语句，每根K线不断的循环。

每次循环中需要做的操作步骤
    1. 更新账户信息
    2. 获取实时数据
    3. 根据最新数据计算买卖信号
    4. 根据目前仓位、买卖信息，结束本次循环，或者进行交易
    5. 交易

# 特别注意的知识点
1. 通过ccxt，在bfx获取margin账户信息，并且交易
2. 实盘产生signal的函数，和回测产生signal的函数不一样
3. 策略执行交易部分，包含6种情况

"""

import ccxt
import time
import pandas as pd
from datetime import datetime, timedelta
from Trade2 import next_run_time
from time import sleep
import json
import requests
import warnings
warnings.filterwarnings("ignore")
pd.set_option('expand_frame_repr', False)  # 当列太多时不换行

# =====初设参数设定
# 基本参数
time_interval = '5m'  # 间隔运行时间，15m代表15分钟，可以改为5m,10m,30m等。注意，不能超过1h
para = [20, 2]  # 策略参数
leverage = 2  # 杠杆倍数，范围(0, 3]。bfx最多支持3.33倍杠杆，但实际当中建议不要超过3。0.5倍杠杆，代表使用50%的保证金进行下单
symbol = 'eth/usd'
ccxt_symbol = 'ETH/USDT'

# 账户设定
api_key = 'Qf4cwyfdSuLwOLlClPo2p1Qdp2qKQlOH7MbmsgetrgX'  # 输入api账户
secret = 'ILrcjnpAM55SMr5eV7B86VqAwoT2MtoNuF7BvSAQ7rK'  # 输入api密码

# 交易symbol设定，由于ccxt的bitfinex获取账户信息时使用的字母不同，例如在获取margin账户USD余额时使用的是USD
# 在使用margin账户下单时使用的是EOS/USDT，position当中的交易对名称为eEOSUSD，因此需要不同的symbol设定
base_coin = symbol.split('/')[-1]  # ‘usd’
base_coin = base_coin.upper()  # ‘USD’
trade_coin = symbol.split('/')[0]  # ‘eth’
trade_symbol = trade_coin.upper() + base_coin  # 'ETHUSD'

# bfx交易所有两个版本的api，第一代和第二代，能用2的时候尽量用2，下单操作用1
# 通过ccxt调用bitfinex，第一代api
exchange = ccxt.bitfinex()
exchange.load_markets()
exchange.apiKey = api_key
exchange.secret = secret

# 通过ccxt调用bitfinex2，第二代api
exchange2 = ccxt.bitfinex2()
exchange2.load_markets()
exchange2.apiKey = api_key
exchange2.secret = secret


# =====常用函数设定
# ===布林线策略
def real_time_signal_bolling(df, para=[100, 2]):
    """
    实盘产生布林线策略信号的函数，和历史回测函数相比，计算速度更快。
    布林线中轨：n天收盘价的移动平均线
    布林线上轨：n天收盘价的移动平均线 + m * n天收盘价的标准差
    布林线上轨：n天收盘价的移动平均线 - m * n天收盘价的标准差
    当收盘价由下向上穿过上轨的时候，做多；然后由上向下穿过下轨的时候，平仓。
    当收盘价由上向下穿过下轨的时候，做空；然后由下向上穿过上轨的时候，平仓。
    :param df:  原始数据
    :param para:  参数，[n, m]
    :return:
    """
    # n代表取平均线和标准差的参数
    # m代表标准差的倍数
    n = para[0]
    print(n)
    m = para[1]

    # ===计算指标
    # 计算均线
    df['median'] = df['close'].rolling(n).mean()  # 此处只计算最后几行的均线值，因为没有加min_period参数
    median = df.iloc[-1]['median']
    # 计算标准差
    df['std'] = df['close'].rolling(n).std(ddof=0)  # ddof代表标准差自由度，只计算最后几行的均线值，因为没有加min_period参数
    std = df.iloc[-1]['std']
    # 计算上轨、下轨道
    upper = median + m * std
    lower = median - m * std

    # ===寻找交易信号
    signal = None
    # 找出做多信号
    if (df.iloc[-1]['close'] > upper) and (df.iloc[-2]['close'] <= upper):
        signal = 1
    # 找出做多平仓信号
    elif (df.iloc[-1]['close'] < median) and (df.iloc[-2]['close'] >= median):
        signal = 0
    # 找出做空信号
    elif (df.iloc[-1]['close'] < lower) and (df.iloc[-2]['close'] >= lower):
        signal = -1
    # 找出做空平仓信号
    elif (df.iloc[-1]['close'] > median) and (df.iloc[-2]['close'] <= median):
        signal = 0
    print('上轨:', upper,' 中轨:', median, ' 下轨：', lower)

    return signal


# ===下单操作
def place_order(exchange, order_type, buy_or_sell, symbol, price, amount):
    """
    下单
    :param exchange: 交易所
    :param order_type: limit, market
    :param buy_or_sell: buy, sell
    :param symbol: 买卖品种
    :param price: 当market订单的时候，price无效
    :param amount: 买卖量
    :return:
    """
    for i in range(5):
        try:
            # 限价单
            if order_type == 'limit':
                # 买
                if buy_or_sell == 'buy':
                    order_info = exchange.create_limit_buy_order(symbol, amount, price, {'type': 'limit'})  # 买单
                # 卖
                elif buy_or_sell == 'sell':
                    order_info = exchange.create_limit_sell_order(symbol, amount, price, {'type': 'limit'})  # 卖单
            # 市价单
            elif order_type == 'market':
                # 买
                if buy_or_sell == 'buy':
                    order_info = exchange.create_market_buy_order(symbol, amount, {'type': 'market'})  # 买单
                # 卖
                elif buy_or_sell == 'sell':
                    order_info = exchange.create_market_sell_order(symbol, amount, {'type': 'market'})  # 卖单
            else:
                pass

            print('下单成功：', order_type, buy_or_sell, symbol, price, amount)
            print('下单信息：', order_info, '\n')
            return order_info

        except Exception as e:
            print('下单报错，1s后重试', e)
            time.sleep(1)

    print('下单报错次数过多，程序终止')
    send_dingding_msg('下单报错次数过多，程序终止')
    exit()


# ===发送钉钉消息，id填上使用的机器人的id
def send_dingding_msg(content, robot_id='f892d66818e9522aac550aa30762548f59fb7f95703fe42055067284eebc85a1'):
    try:
        msg = {
            "msgtype": "text",
            "text": {"content": content + '\n' + datetime.now().strftime("%m-%d %H:%M:%S")}}
        headers = {"Content-Type": "application/json;charset=utf-8"}
        url = 'https://oapi.dingtalk.com/robot/send?access_token=' + robot_id
        body = json.dumps(msg)
        requests.post(url, data=body, headers=headers)
        print('成功发送钉钉')
    except Exception as e:
        print("发送钉钉失败:", e)


# ===获取bfx交易所margin账户和仓位信息
def fetch_account_info(exchange, base_coin, trade_symbol):
    """
    获取账户信息
    :param exchange: 交易所
    :param base_coin: 基准币的名称
    :param trade_symbol: 交易对的名称
    :return:
    :param account_info: dict形式，包含账户和仓位信息
    """
    # 创建用于存放账户信息的变量
    account_info = {'仓位数量': 0, '仓位成本': 0, '仓位利润': 0, '仓位爆仓价格': 0}

    while True:

        # 获取账户的资产信息
        try:
            data = exchange.private_post_auth_r_wallets()  # 从bfx交易所获取账户balance信息
        except Exception as e:
            send_dingding_msg('获取账户信息失败')
            print(e)
            continue
        data = pd.DataFrame(data, columns=['交易账户', '币种', '数量', 'unknow', 'unknow2'])  # 将数据转化为df格式
        condition1 = data['交易账户'] == 'margin'
        condition2 = data['币种'] == base_coin
        account_info['账户保证金'] = float(data.loc[condition1 & condition2, '数量'])

        # 获取账户的margin持仓信息
        try:
            position_info = exchange.private_post_auth_r_positions()  # 从bfx交易所获取账户的持仓信息
        except Exception as e:
            send_dingding_msg('获取持仓信息失败')
            print(e)
            continue
        print(position_info)
        if len(position_info) > 0:  # 当持仓信息信息不为空时
            position_info = pd.DataFrame(position_info, columns=['交易对', '状态', '持仓量', '成本价格', '借币利息',
                                                                 'unknow1', '利润', 'unknow2', '爆仓价格',
                                                                 'unknow3'])  # 将数据转化为df格式
            condition1 = position_info['交易对'] == ('t' + str(trade_symbol))
            position_info = position_info.loc[condition1, :]
            if len(position_info) > 0:
                account_info['仓位数量'] = float(position_info.iloc[0]['持仓量'])
                account_info['仓位成本'] = float(position_info.iloc[0]['成本价格'])
                account_info['仓位利润'] = float(position_info.iloc[0]['利润'])
                account_info['仓位爆仓价格'] = float(position_info.iloc[0]['爆仓价格'])

        break

    return account_info


# ===获取bitfinex交易所k线
def get_bitfinex_candle_data(exchange, symbol, time_interval, limit):
    while True:
        try:
            content = exchange.fetch_ohlcv(symbol=symbol, timeframe=time_interval, limit=limit)
            break
        except Exception as e:
            send_dingding_msg(content='抓不到k线，稍等重试')
            print(e)
            sleep(5 * 1)

    df = pd.DataFrame(content, dtype=float)
    df.rename(columns={0: 'MTS', 1: 'open', 2: 'high', 3: 'low', 4: 'close', 5: 'volume'}, inplace=True)
    df['candle_begin_time'] = pd.to_datetime(df['MTS'], unit='ms')
    df['candle_begin_time_GMT8'] = df['candle_begin_time'] + timedelta(hours=8)
    df = df[['candle_begin_time_GMT8', 'open', 'high', 'low', 'close', 'volume']]
    # 在这里使用的是中国本地时间 所以需要GMT8 如果在服务器上跑直接使用candle_begin_time这一列就可以了
    # df = df[['candle_begin_time', 'open', 'high', 'low', 'close', 'volume']]
    return df


# =====主函数
def main():
    # ===钉钉内容
    msg_content = '策略报告：'
    sleep(3 * 1)

    # ===获取现在的账户信息和仓位信息
    account_info = fetch_account_info(exchange2, base_coin, trade_symbol)
    print(account_info)
    msg_content += '\n账户持有美元:' + str(account_info['账户保证金'])
    msg_content += '\n持有' + str(trade_coin) + ':' + str(account_info['仓位数量'])
    if account_info['仓位数量'] != 0:
        msg_content += '\n仓位成本价:' + str(account_info['仓位成本'])
        msg_content += '\n仓位利润:' + str(account_info['仓位利润'])
        msg_content += '\n仓位爆仓价:' + str(account_info['仓位爆仓价格'])

    # ===sleep直至需要运行的时间
    run_time = next_run_time(time_interval)
    sleep(max(0, (run_time - datetime.now()).seconds))
    while True:  # 在靠近目标时间时
        if datetime.now() < run_time:
            continue
        else:
            break
    # ===开始运行，获取最新数据
    while True:
        # 获取数据
        df = get_bitfinex_candle_data(exchange2, ccxt_symbol, time_interval,
                                      limit=max(para) + 5)  # 这里使用的是课程里的代码，获取的是bitfinex的数据
        # 判断是否包含最新的数据
        _temp = df[df['candle_begin_time_GMT8'] == (run_time - timedelta(minutes=int(time_interval.strip('m'))))]
        if _temp.empty:
            print('获取数据不包含最新的数据，重新获取')
            send_dingding_msg('获取不到最新数据，重新获取')
            sleep(3 * 1)
            continue
        else:
            now_price = df.iloc[-1]['close']  # now_price 为现在该币种的价格
            df = df[df['candle_begin_time_GMT8'] < pd.to_datetime(run_time)]  # 去除target_time周期的数据
            print(df.tail(5))
            break
    msg_content += '\n' + str(trade_coin) + '实时价格:' + str(now_price)

    # ===计算交易信号
    signal = real_time_signal_bolling(df, para=para)
    print('本周期交易信号:', signal)
    msg_content += '\n本周期交易信号:' + str(signal)


    # ===判断交易方向并且下单，除了无交易信号之外，总共有6种情况
    # 之前为空单，现在要平仓的情况
    if account_info['仓位数量'] < 0 and signal == 0:
        trade_amount = abs(account_info['仓位数量'])
        print('\n关闭空单')
        place_order(exchange=exchange, order_type='limit', buy_or_sell='buy', price=now_price * 1.002,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n_关闭空单_'
        msg_content += '\n交易数量:' + str(trade_amount) + '\n'

    # 之前为多单，现在要平仓的情况
    elif account_info['仓位数量'] > 0 and signal == 0:
        trade_amount = abs(account_info['仓位数量'])
        print('\n关闭多单')
        place_order(exchange=exchange, order_type='limit', buy_or_sell='sell', price=now_price * 0.998,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n_关闭多单_'
        msg_content += '\n交易数量:' + str(trade_amount) + '\n'

    # 之前没有仓位，现在要开多单的情况
    elif account_info['仓位数量'] == 0 and signal == 1:
        print('\n开多单')
        trade_amount = abs(account_info['账户保证金'] * leverage / now_price)  # 计算开仓量
        place_order(exchange=exchange, order_type='limit', buy_or_sell='buy', price=now_price * 1.002,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n_开多单_'
        msg_content += '\n交易数量:' + str(trade_amount) + '\n'

    # 之前空仓 现在要开空单的情况
    elif account_info['仓位数量'] == 0 and signal == -1:
        print('\n开空单')
        trade_amount = abs(account_info['账户保证金'] * leverage / now_price)  # 计算开仓量
        place_order(exchange=exchange, order_type='limit', buy_or_sell='sell', price=now_price * 0.998,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n_开空单_'
        msg_content += '\n交易数量:' + str(trade_amount) + '\n'

    # 之前开多单现在要开空单的情况
    elif account_info['仓位数量'] > 0 and signal == -1:
        print('\n关闭多单并开空单')
        # 平多
        trade_amount = abs(account_info['仓位数量'])
        place_order(exchange=exchange, order_type='limit', buy_or_sell='sell', price=now_price * 0.998,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        # 更新账户信息
        account_info = fetch_account_info(exchange2, base_coin, trade_symbol)
        # 开空
        trade_amount = abs(account_info['账户保证金'] * leverage / now_price)  # 计算开仓量
        place_order(exchange=exchange, order_type='limit', buy_or_sell='sell', price=now_price * 0.998,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n关闭多单并开空单'
        msg_content += '\n开空交易数量:' + str(trade_amount) + '\n'

    # 之前开空单现在要开多单的情况
    elif account_info['仓位数量'] < 0 and signal == 1:
        print('\n关闭空单并开多单')
        # 平空
        trade_amount = abs(account_info['仓位数量'])
        place_order(exchange=exchange, order_type='limit', buy_or_sell='buy', price=now_price * 1.002,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        # 更新账户信息
        account_info = fetch_account_info(exchange2, base_coin, trade_symbol)
        # 开多
        trade_amount = abs(account_info['账户保证金'] * leverage / now_price)  # 计算开仓量
        place_order(exchange=exchange, order_type='limit', buy_or_sell='buy', price=now_price * 1.002,
                    symbol=ccxt_symbol,
                    amount=trade_amount)
        msg_content += '\n关闭空单并开多单'
        msg_content += '\n开多交易数量:' + str(trade_amount) + '\n'

    # 无交易信号
    else:
        msg_content += '\n本周期无操作'
        print('本周期无操作')

    # ===本周起运行结束，发送钉钉

    sleep(10 * 1)
    try:
        send_dingding_msg(msg_content)
    except Exception as e:
        print(e)

    # =====本次交易结束
    print('=====本次运行完毕\n')
    sleep(30 * 1)


#  =====运行主体
while True:
    try:
        main()
        time.sleep(10)
    except Exception as e:
        send_dingding_msg('系统出错，10s之后重新运行')
        print(e)
        time.sleep(10)
