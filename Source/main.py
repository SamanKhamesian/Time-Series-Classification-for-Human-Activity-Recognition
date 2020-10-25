import Source.cnn_lstm
import Source.lstm


def runner():
    print('LSTM network is running ...')
    Source.lstm.run()
    print('LSTM network is finished!')
    print('------------------------------------')
    print('Now, CNN-LSTM network is running ...')
    Source.cnn_lstm.run()
    print('CNN-LSTM network is finished!')


if __name__ == '__main__':
    runner()
