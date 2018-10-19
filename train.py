#!/usr/bin/env python
from collections import defaultdict
import torch
import numpy as np
from tqdm import tqdm  # プログレスバー表示用

# GPUを使うならTrue
gpu = False


class Encoder(torch.nn.Module):
    def __init__(self, n_source_vocab, embedding_dim, hidden_dim):
        """

        :param n_source_vocab: ソース言語の語彙数
        :param embedding_dim: 単語の埋め込み(ベクトル表現)の次元数
        :param hidden_dim: LSTMの状態ベクトルの次元数
        """
        super(Encoder, self).__init__()

        # 単語をベクトルに変換するModule
        # padding_idxを指定すると埋め込みの値が0に固定される。
        self.embedding = torch.nn.Embedding(n_source_vocab, embedding_dim, padding_idx=1)
        self.rnn = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)

    def forward(self, xs):
        """
        系列のバッチを固定次元ベクトルにエンコードする
        :param xs: 系列データのリスト(ミニバッチ)。長さ降順にソートされている
        :return: LSTMの状態(h,c) h.shape == c.shape == (1, batch_size, hidden_dim)
        """
        # xsをPackedSequenceに変換
        xs = torch.nn.utils.rnn.pack_sequence(xs)

        # PackedSequenceの状態で単語を埋め込む
        exs = torch.nn.utils.rnn.PackedSequence(self.embedding(xs.data), xs.batch_sizes)

        # RNNに系列を通す。状態にNoneを渡すとは自動的に0埋めされたベクトルが用意される
        out_seq, state = self.rnn(exs, None)
        return state


class Decoder(torch.nn.Module):
    def __init__(self, target_vocab_size, embedding_dim, hidden_dim, sos=2):
        super(Decoder, self).__init__()
        self.embedding = torch.nn.Embedding(target_vocab_size, embedding_dim, padding_idx=1)  # Encoderと同様
        self.rnn = torch.nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim)  # Encoderと同様
        self.W = torch.nn.Linear(hidden_dim, target_vocab_size)  # RNNの出力を語彙数次元のベクトルに変換する行列
        self.sos = sos  # <sos>のindex

    def forward(self, state, ys):
        """
        エンコードされた状態から出力系列のスコアを出力
        TeacherForcing
        :param state: エンコーダのLSTMの最終状態
        :param ys: paddingされた正解出力系列
        :return:
        """
        eys = self.embedding(ys)
        os, state = self.rnn(eys, state)

        # 出力系列を語彙数次元ベクトルに変換
        # この値をSoftMaxしたものが単語の出現する確率になる
        os_v = self.W(os)
        return os_v

    def predict(self, state, max_length):
        """
        エンコード状態だけから出力を推論する
        :param state: Encoderの出力
        :param max_length: 最大出力系列長
        :return:
        """
        batch_size = state[0].shape[1]
        device = state[0].device  # GPU対応

        # デコード開始の<sos>
        ys = self.embedding(
            torch.full((batch_size,), self.sos).long().to(device)
        )

        result = []  # 各ステップの出力単語が入る
        for i in range(max_length):
            # ysを長さ1の系列としてDecodeする
            # そのステップの出力(長さ1)が出力され、状態が更新される
            seq, state = self.rnn(ys[None], state)

            # 出力を語彙数次元ベクトルに変換
            # SoftMaxして確率値とみなしても良いが、最大値を探すだけなのでSoftMaxを省略している(結果は変わらない)
            p = self.W(seq[0])
            ids = torch.argmax(p, dim=1)

            # 出力単語を次の入力とする
            ys = self.embedding(ids)
            result.append(ids)

        # 出力単語リストを転置してバッチ方向のリストに戻しておく
        return torch.stack(result).transpose(0, 1).cpu().numpy()


class Seq2Seq(torch.nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, embedding_dim, hidden_dim):
        """

        :param source_vocab_size: ソース言語の語彙数
        :param target_vocab_size: ターゲット言語の語彙数
        :param embedding_dim: 単語の埋め込み(ベクトル表現)の次元数
        :param hidden_dim: LSTMの状態ベクトルの次元数
        """
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(source_vocab_size, embedding_dim, hidden_dim)
        self.decoder = Decoder(target_vocab_size, embedding_dim, hidden_dim)

    def forward(self, xs, ys):
        """
        誤差を計算する
        :param xs: 入力系列のリスト(ミニバッチ)
        :param ys: 正解系列のリスト(ミニバッチ)。先頭に<sos>、末尾に<eos>が挿入されている
        :return:
        """
        # 入力をエンコードする
        state = self.encoder(xs)

        # paddingする。<pad>のindexは1であることに注意
        ys = torch.nn.utils.rnn.pad_sequence(ys, padding_value=1)

        # デコード系列を得る。shapeは(seq_length, batch_size, vocab_size)
        os_v = self.decoder(state, ys)

        # 出力系列の末尾と、正解系列の先頭を削除してそれぞれflattenする
        # flattenするのはまとめてcross_entropyを計算するため
        out = os_v[:-1].view(-1, os_v.shape[-1])
        target = ys[1:].view(-1)

        # ignore_indexを指定して、paddingを入力したときの誤差を無視している
        loss = torch.nn.functional.cross_entropy(out, target, ignore_index=1)
        return loss

    def translate(self, xs, max_length):
        """
        翻訳する
        :param xs: 入力系列のリスト
        :param max_length: 出力最大長
        :return:
        """
        batch_size = len(xs)

        # エンコード/デコード
        state = self.encoder(xs)
        output = self.decoder.predict(state, max_length=max_length)

        # 出力系列の<eos>以降を削除する
        result = []
        for i in range(batch_size):
            d = list(output[i])
            try:
                eos_index = d.index(3)  # <eos>のindexを探す
                d = d[:eos_index]
            except:
                pass
            result.append(d)
        return result


def build_vocab(sentences):
    itos = ["<unk>", "<pad>", "<sos>", "<eos>"]  # id から文字列へのmap。特殊トークンは最初に入れておく
    stoi = defaultdict(lambda: 0)  # 文字列からidへのmap
    for i, v in enumerate(itos):
        stoi[v] = i
    for sentence in sentences:
        for token in sentence:
            if token not in stoi:  # まだ stoiに入ってない単語を見つけたらそれを追加する
                stoi[token] = len(itos)
                itos.append(token)
    return stoi, itos


def numericalize(sentences, stoi):
    """
    単語列を数値化する
    :param sentences:
    :param stoi:
    :return:
    """
    return [
        [stoi[token] for token in sentence]
        for sentence in sentences
    ]


def to_string(sentences, itos, join_token):
    """
    数値列を文字列に復元する
    :param sentences:
    :param itos:
    :param join_token: 単語の間に挟む文字列。enなら" "、jaなら""
    :return:
    """
    return [
        join_token.join([
            itos[token] for token in sent
        ]) for sent in sentences
    ]


def load_pretrained_model():
    model = Seq2Seq(6638, 8778, 256, 256)
    model.load_state_dict(torch.load("pretrained_params", map_location="cpu"))
    return model


if __name__ == '__main__':

    with open("small_parallel_enja/train.en") as f:
        en_train = [line[:-1].split() for line in f]  # 末尾の改行は削除している
    with open("small_parallel_enja/test.en") as f:
        en_test = [line[:-1].split() for line in f]

    with open("small_parallel_enja/train.ja") as f:
        ja_train = [line[:-1].split() for line in f]
    with open("small_parallel_enja/test.ja") as f:
        ja_test = [line[:-1].split() for line in f]

    # 語彙の構築
    en_stoi, en_itos = build_vocab(en_train)
    ja_stoi, ja_itos = build_vocab(ja_train)
    src_vocab_size = len(en_itos)
    target_vocab_size = len(ja_itos)

    # 単語の数値化
    en_train_numericalized = numericalize(en_train, en_stoi)
    ja_train_numericalized = numericalize(ja_train, ja_stoi)
    en_test_numericalized = numericalize(en_test, en_stoi)
    ja_test_numericalized = numericalize(ja_test, ja_stoi)

    train_data = list(zip(en_train_numericalized, ja_train_numericalized))
    test_data = list(zip(en_test_numericalized, ja_test_numericalized))[:10]  # 結果を見るだけなので10個だけ使う

    batch_size = 64
    embedding_dim = 64
    hidden_dim = 64

    # モデルの構築
    model = Seq2Seq(src_vocab_size, target_vocab_size, embedding_dim, hidden_dim)
    # model = load_pretrained_model() # 学習済みモデルのロード
    if gpu:  # GPU対応
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters())


    def show_result():
        """
        テストデータの翻訳結果を出力
        :return:
        """
        model.train(False)
        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                batch = test_data[i:i + batch_size]
                xs = [torch.tensor(x) for x, y in batch]
                ys = [y for x, y in batch]
                if gpu:
                    xs = [x.cuda() for x in xs]
                xs_length = np.array([len(x) for x in xs])
                sorted_index = np.argsort(-xs_length)
                sorted_xs = [xs[i] for i in sorted_index]
                sorted_ys = [ys[i] for i in sorted_index]
                result = model.translate(sorted_xs, 100)

                string_xs = to_string(sorted_xs, en_itos, " ")
                string_result = to_string(result, ja_itos, "")
                string_ys = to_string(sorted_ys, ja_itos, "")
                for x, y, t in zip(string_xs, string_result, string_ys):
                    print("\ninput:  {}\noutput: {}\ntarget: {}".format(x, y, t))

        model.train(True)


    show_result()
    for ep in range(1, 10):
        for i in tqdm(range(0, len(train_data), batch_size), desc="training epoch: {}".format(ep)):
            batch = train_data[i:i + batch_size]

            # データのtensor化
            xs = [torch.tensor(x) for x, y in batch]
            ys = [torch.tensor([2] + y + [3]) for x, y in batch]  # <sos>,<eos>の挿入
            if gpu:
                xs = [x.cuda() for x in xs]
                ys = [y.cuda() for y in ys]

            # 長さ降順にソートする
            xs_length = np.array([len(x) for x in xs])
            sorted_index = np.argsort(-xs_length)
            sorted_xs = [xs[i] for i in sorted_index]
            sorted_ys = [ys[i] for i in sorted_index]

            # 学習
            optimizer.zero_grad()
            loss = model(sorted_xs, sorted_ys)
            loss.backward()
            optimizer.step()

        show_result()
    # torch.save(model.state_dict(), "trained_model")  # 学習したモデルの保存


# -------事前学習済みモデル出力-------
# input:  i 'm a person who lives for the moment .
# output: 今のところ住んでいる人がいる。
# target: 私は<unk>的な生き方をしている人間です。
#
# input:  will you give me your reasons for doing this ?
# output: これを教えてくれるな。
# target: こんなことをした理由を言いなさい。
#
# input:  he is no less kind than his sister .
# output: 彼は姉と妹に劣らない。
# target: 彼はお姉さんに劣らず親切だ。
#
# input:  i 'm about to tell you the answer .
# output: あなたに答えるのを楽しみにしておいて。
# target: あなたに返事をしようとしているところです。
#
# input:  they finally acknowledged it as true .
# output: 彼らはそれを真実でした。
# target: 彼らはついにそれが真実だと認めた。
#
# input:  he didn 't care for swimming .
# output: 彼は、たばこを吸わないと言った。
# target: 彼は水泳が得意ではなかった。
#
# input:  you must be back before ten .
# output: １０時前には戻らなければならない。
# target: １０時前に戻らなければならない。
#
# input:  she lives next door to us .
# output: 彼女は私たちの隣に住んでいる。
# target: 彼女は私たちの隣の家にすんでいる。
#
# input:  we have this game on ice .
# output: 私たちはこのゲームに勝つ。
# target: この試合はいただきだ。
#
# input:  break a leg .
# output: 動物を取って。
# target: 成功を祈るわ。
