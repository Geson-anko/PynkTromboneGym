# PynkTromboneGym
The vocal tract environment for speech generations by reinforcement learning.  
See [pynktrombone](https://github.com/Geson-anko/pynktrombone) and [Original PinkTrombone](https://imaginary.github.io/pink-trombone/).

# Installation
```sh
pip install git+https://github.com/Geson-anko/PynkTromboneGym.git@master
```
Or, clone this repository, and run a following command.
```sh
pip install -e .
```

# Sample Code
See `sample.py`


# Environment Definition
PynkTromboneEnvは人間のVocal Tractをシミュレーションし、強化学習の枠組みを用いて音声生成タスクを定義するためのEnvironmentです.  
基本的にGym APIに従っています。

## Construction (`__init__`)
環境を構築します。いくつか指定する必要のある項目（引数）が存在します。
- target_sound_files: Iterable[str]  
    模倣対象となる音声ファイル群を渡します。`wav`形式が望ましいですが、ffmpegをインストールしている場合は`mp3`などの形式も読み込む事ができます。
- sample_rate: float  
    生成する音声の解像度です。ターゲットとなる音声もこのサンプリングレートに合わせられます。
- default_frequency: float  
    基準となる声の高さです。この周波数から $\pm{1}$ オクターブが生成可能な周波数帯です。

- generate_chunk: int  
    1ステップで生成する音声波形の長さです。pynkTromboneではデフォルトで`1024`です。  
    ちなみに `generate_chunk / sample_rate` [s] が1ステップで生成する波形の時間間隔です。
- stft_window_size: int  
    波形をstftする時のウィンドウサイズです。デフォルトでは`1024`です。  

- stft_hop_length: int  
    波形をstftする時のホップ幅です。デフォルトは`stft_window_size/4`を使用します。

- rendering_figure_size: tuple[float ,float]   
    `render`メソッドでレンダリングされる画像の大きさです。`matplotlib`を使用しているため(width, height)の順で、単位は*Inch*です。
- set_target_sound_files(file_paths: Sequence[str]) : method  
    このメソッドを使用することで使用する音声ファイル群を変更する事ができます。


### Reset
環境をリセットします。ランダムに音声ファイルを選択しターゲット音声とします。内部のVocal Tract モデルもリセットします。
返り値として、初期観測を返します。

## Observation
音声生成を強化学習の枠組みで行うために必要な観測情報を定義します。これらはOpenAI GymのAPIを用いて`Dict`型で返されます。これらの値名は`pynktrombonegym.spaces.ObservationSpaceNames`クラスに記述されています。

- target_sound_wave  
    次のステップで生成する波形です。
- generated_sound_wave  
    そのステップで生成した波形です。
- target_sound_spectrogram    
    次のステップで生成する波形のスペクトログラムです。

- generated_sound_spectrogram  
    そのステップで生成した波形のスペクトログラムです。報酬計算に使用されます。
- frequency, pitch_shift  
    現在の声帯の周波数と、`default_frequency`からピッチシフトした大きさ（指数部）を返します。
- tenseness  
    現在の声帯の掠れ具合の値です。`<Env>.voc.tenseness`からもアクセスする事ができます。
- current_tract_diameters  
    円筒列で近似した声道の現在の直径の値を配列で返します。
- nose_diameters  
    円筒列で近似した鼻腔の現在の直径の値を配列で返します。

## Action
この環境モデルの行動を定義します。行動は全てNumpyArray形式で、行動空間は`Dict`型で定義されます。各行動の名前は`pynktrombonegym.spaces.ActionSpaceNames`クラスに記述されています。

- Glottis  
    声門を調整します。
    - pitch_shift  
        Range: [-1, 1] (この値は環境をラップすることで変更できます。)  
        `default_frequency`からどれだけピッチシフトをする値です。周波数は次の式で与えられます。  
        
        $$ frequency = default \ frequency \times  2^{pitch \ shift} $$

    - tenseness  
        Range: [0, 1]  
        声の掠れ具合です。  
- tract_diameters  
    [詳細はPynkTromboneのREADME.mdを参照願います。](https://github.com/Geson-anko/pynktrombone)
    - trachea  
        Range: [0, 3.5]  
    - epiglottis  
        Range: [0, 3.5]  
    - velum  
        Range: [0, 3.5]  
    - tongue_index  
        Range: [12.0, 40.0]  
        Note: この値はfloat型です。詳しくは[`PynkTrombone`の実装を参照ください](https://github.com/Geson-anko/pynktrombone/blob/master/pynktrombone/voc.py#L201)
    - tongue_diameter  
        Range: [0, 3.5]
    - lips  
        Range: [0, 1.5]

### Step
上記のアクションは、全て`pynktrombone.voc.Voc`の調整のみに使われます。  
`Voc.play_chunk()`によって音声波形を生成し、観測、報酬、終了判定、デバッグ情報を返します。  

#### About `done`
ターゲットとなる音声の長さに生成波形が達した場合、`done`となります。この状態で行動しようとすると例外を返します。  

## Reward  
target_sound_spectrogramとgenerated_sound_spectrogramの平均二乗誤差の符号を反転させた値を返します。


## Visualize (Render)
`current_tract_diameters`と`nose_diameters`をplotした画像を返します。  
次のような形でプロットされます。
![sample_render](data/sample_render.png)

## Wrappers
PynkTrombone Environmentクラスをラップするいくつかのクラスがあります。
### Log1pMelSpectrogram
stftによって生成されたスペクトログラムをメル周波数スペクトログラムにした後、対数スケールに変換する派生クラスです。   
Note: 実際はラッパーではなく、継承したSubclassであることに注意してください。`__init__`を呼び出す際の引数がいくつか追加されています。  
- mel_channels: int  
    Melスケールにする時のチャネル数です。デフォルト値は`80`です。
- dtype: Any  
    Mel filter bankの型です。デフォルト値は`np.float32`です。

次のようにして使用します。  

```py
from pynktrombonegym.wrappers.log1p_mel_spectrogram import Log1pMelSpectrogram

env = Log1pMelSpectrogram(target_sound_files=..., mel_channels=64, ...)
```

## ActionByAcceleration
これは物理系に従い、加速度を入力に取るようにラップする`ActionWrapper`です。  
ランダム方策による生成結果がより自然な形となり、学習が用意になることが予想されます。   
加速度に変換された行動の範囲は0が中心になります。  

次のようにして使うことができます。  

```py
from pynktrombonegym.env import PynkTrombone
from pynktrombonegym.wrappers.action_by_acceleration import ActionByAcceleration

env = PynkTrombone(target_sound_files=...)
wrapped = ActionByAcceleration(env, action_scaler=env.generate_chunk/env.sample_rate)
```

このラッパーは以下の引数を持ちます。  
- env: gym.Env  
    PynkTrombone環境モデルのインスタンスです。
- action_scaler: float  
    行動の値の範囲をこの値でスケールします。`generate_chunk/sample_rate`を基準とすると良いでしょう。
- initial_pos: Optional[Dict]  
    PynkTrombone環境モデルの初期行動です。指定されない場合はランダムにサンプルされます。   
- ignore_actions: Optional[Iterable[str]]  
    ラップしない行動の種類を文字列で指定します。  


# References
- https://www.imaginary.org/program/pink-trombone  
- [Voc: A vocal tract physical model implementation.](https://pbat.ch/res/voc/voc.pdf)

