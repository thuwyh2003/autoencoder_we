import pandas as pd
import numpy as np
from utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')


def calculate_R2(model, type, input=None, complete_r=None):
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    oos_ret = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)]

    if not isinstance(input, np.ndarray):
        print('type: ', type)
        if isinstance(model, str):
            output_path = f'results/{type}/{model}_{type}.csv'
        else:
            output_path = f'results/{type}/{model.name}_{type}.csv'
        print('path : ', output_path)
        model_output = pd.read_csv(output_path)
    else:
        model_output = input
        model_output = pd.DataFrame(model_output, columns=CHARAS_LIST)
        model_output['DATE'] = oos_ret['DATE'].to_list()
  
    for col in model_output.columns: # hard code for format error
        model_output[col] = model_output[col].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))
    
    residual_square = ((oos_ret.set_index('DATE') - model_output.set_index('DATE'))**2).dropna()
    residual_square = (1 - (residual_square == np.inf) * 1.0) * residual_square # drop Inf outliers
    
    total_square = oos_ret.set_index('DATE')**2
    total_square = (1 - (total_square == np.inf) * 1.0) * total_square # drop Inf outliers
    
    model_output_R2 = 1 - np.sum(residual_square.values)/np.sum(total_square.values)
    
    if not isinstance(input, np.ndarray):
        return model_output_R2
    
    else:
        no_omit_output = complete_r
        no_omit_output = pd.DataFrame(no_omit_output, columns=CHARAS_LIST)
        no_omit_output['DATE'] = oos_ret['DATE'].to_list()
        
        no_omit_residual_square = ((oos_ret.set_index('DATE') - no_omit_output.set_index('DATE'))**2).dropna()
        no_omit_residual_square = (1 - (no_omit_residual_square == np.inf) * 1.0) * no_omit_residual_square # drop Inf outliers
        
        no_omit_model_output_R2 = 1 - np.sum(no_omit_residual_square.values)/np.sum(total_square.values)
        
        return no_omit_model_output_R2 - model_output_R2 # the difference of R^2, i.e. the importance of characteristics



def alpha_plot(model, type, save_dir='imgs'):
    if 'alpha' not in os.listdir(save_dir):
        os.mkdir(f'{save_dir}/alpha')
    
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    oos_result = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)].set_index('DATE')
    print(OOS_start, OOS_end)
    output_path = f'results/{type}/{model.name}_{type}.csv'
    inference_result = pd.read_csv(output_path)
    inference_result = inference_result.set_index('DATE')
    # print(inference_result)
    pricing_error_analysis = []
    for col in CHARAS_LIST:
        raw_return = oos_result[col].mean()
        # print(oos_result[col],inference_result[col])
        # breakpoint()
        error = oos_result[col] - inference_result[col]
        alpha = error.mean()
        t_stat = abs(error.mean()/error.std()) * np.sqrt(oos_result.shape[0])
        pricing_error_analysis.append([raw_return, alpha, t_stat])

    pricing_error_analysis = pd.DataFrame(pricing_error_analysis, columns = ['raw ret', 'alpha', 't_stat'], index=CHARAS_LIST)
    # print(pricing_error_analysis)
    lower_point = min(np.min(pricing_error_analysis['raw ret']), np.min(pricing_error_analysis['alpha'])) * 1.15
    upper_point = max(np.max(pricing_error_analysis['raw ret']), np.max(pricing_error_analysis['alpha'])) * 1.15

    significant_mask = pricing_error_analysis['t_stat'] > 3

    plt.scatter(pricing_error_analysis.loc[significant_mask]['raw ret'], pricing_error_analysis.loc[significant_mask]['alpha'], marker='^', color='r', alpha=0.6, label=f'#Alphas(|t|>3.0)={np.sum(significant_mask*1.0)}')
    plt.scatter(pricing_error_analysis.loc[~significant_mask]['raw ret'], pricing_error_analysis.loc[~significant_mask]['alpha'], marker='o', color='b', alpha=0.6, label=f'#Alphas(|t|<3.0)={94-np.sum(significant_mask*1.0)}')
    plt.plot(np.linspace(lower_point, upper_point, 10), np.linspace(lower_point, upper_point, 10), color='black')

    plt.ylabel('Alpha (%)')
    plt.xlabel('Raw Return (%)')
    plt.legend()

    plt.title(model.name)
    plt.savefig(f'{save_dir}/alpha/{model.name}_inference_alpha_plot.png')
    plt.close()
    

def plot_R2_bar(R_df, type):
    
    R_df['Model'] = R_df[0].apply(lambda x: x.split('_')[0])

    labels = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5']
    FF = (R_df.loc[R_df['Model']=='FF'][1]*100).to_list()
    PCA = (R_df.loc[R_df['Model']=='PCA'][1]*100).to_list()
    IPCA = (R_df.loc[R_df['Model']=='IPCA'][1]*100).to_list()
    CA0 = (R_df.loc[R_df['Model']=='CA0'][1]*100).to_list()
    CA1 = (R_df.loc[R_df['Model']=='CA1'][1]*100).to_list()
    CA2 = (R_df.loc[R_df['Model']=='CA2'][1]*100).to_list()
    CA3 = (R_df.loc[R_df['Model']=='CA3'][1]*100).to_list()
    # VAE = (R_df.loc[R_df['Model']=='VAE'][1]*100).to_list()

    x = np.arange(len(labels))  # 标签位置
    width = 0.11

    fig, ax = plt.subplots(figsize=(15, 5))
    # ax.bar(x - width*3 , FF, width, label='FF', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[1]))
    # ax.bar(x - width*2 , PCA, width, label='PCA', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[2]))
    # ax.bar(x - width , IPCA, width, label='IPCA', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[3]))
    ax.bar(x + 0.00, CA0, width, label='CA0', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[4]))
    ax.bar(x + width , CA1, width, label='CA1', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[5]))
    ax.bar(x + width*2 , CA2, width, label='CA2', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[6]))
    ax.bar(x + width*3 , CA3, width, label='CA3', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))
    # ax.bar(x + width*4 , VAE, width, label='CA3', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))

    ax.set_ylabel(f'Portfolio {type} R^2 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig(f'imgs/{type}_R2.png')
    plt.close()



def plot_R2_table(R_df, type):
    plt.figure(dpi=200)
    
    for col in R_df.columns:
        R_df[col] = R_df[col].apply(lambda x: round_number(x))

    R_df = R_df.reset_index()
    R_df.columns = ['Model', 'K=1', 'K=2', 'K=3', 'K=4', 'K=5']


    fig_total =  ff.create_table(R_df,
                        colorscale=[[0, 'white'],
                                    [0.01, 'lightgrey'],
                                    [1.0, 'white']],
                        font_colors=['#000000', '#000000',
                                    '#000000'])
    fig_total.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    fig_total.write_image(f"imgs/R2_{type}_table.png", scale=4)
    


def round_number(num):
    num = str(round(num*100, 2))
    while len(num.split('.')[1]) < 2:
        num = num + '0'
    return num


    
if __name__=="__main__":
    CAs = ["CA0_1", "CA0_2", "CA0_3", "CA0_4", "CA0_5",  "CA1_1", "CA1_2", "CA1_3", "CA1_4", "CA1_5", "CA2_1", "CA2_2", "CA2_3", "CA2_4", "CA2_5",  "CA3_1", "CA3_2", "CA3_3", "CA3_4", "CA3_5"]
    # FFs = ["FF_1", "FF_2", "FF_3", "FF_4", "FF_5", "FF_6"]
    # PCAs = ["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5", "PCA_6"]
    # IPCAs = ["IPCA_1", "IPCA_2", "IPCA_3", "IPCA_4", "IPCA_5", "IPCA_6"]
    models = CAs
    
    ## Plot R^2 bars
    total_R2 = []
    for m in models:
        m=m[0:3]+'_wyh'+m[3:]
        total_R2.append(calculate_R2(m, 'inference'))
    R_total = pd.DataFrame([models, total_R2]).T

    predict_R2 = []
    for m in models:
        m=m[0:3]+'_wyh'+m[3:]
        predict_R2.append(calculate_R2(m, 'predict'))
    R_pred = pd.DataFrame([models, predict_R2]).T
    print(R_pred)
    plot_R2_bar(R_total, 'total')
    plot_R2_bar(R_pred, 'pred')
    
    ## Save R^2 tables
    R_total_df = pd.DataFrame(np.array(total_R2).reshape(-1, 5), columns = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5'], index=['CA0', 'CA1', 'CA2', 'CA3'])
    R_pred_df = pd.DataFrame(np.array(predict_R2).reshape(-1, 5), columns = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5'], index=['CA0', 'CA1', 'CA2', 'CA3'])
    
    plot_R2_table(R_total_df, 'total')
    plot_R2_table(R_pred_df, 'pred')  
    
    
    ## Plot characteristics importance heatmap
    models = ["CA0_5", "CA1_5", "CA2_5", "CA3_5"]
    #TODO: paste results from R_squares/
    # R2_omit = []
    R2_omit=[ -0.0027801188524286813, -0.0011279748030672554, -0.003315564540722349, 0.0008969755921546252, 0.0025269355599559784, 0.0002643508785096227, 0.0017792010860717689, -0.0010888924045997506, -0.0059249349976278065, -0.0029743167233856616, 0.00020633761284405416, 0.002228200803396918, 0.00038057430809956827, 0.0007937745732230495, -0.0009128014385932914, -0.003560718454214351, -0.0009983028063342836, -0.004919888791146687, -0.0021592986202261244, -0.001953365200970225, -0.0036228333312043892, 0.0005655232635990437, -0.0013165260531020628, -0.0002230204653820289, -7.60859289519189e-05, -0.00044983326771330656, -0.0018830891626466784, -0.00023060828118148802, -1.9705120466673165e-05, 0.00269725982660729, 0.0003923273140715722, -3.4631427193843045e-05, 0.0034201278944374636, -0.003920165170175238, -7.363286432182647e-05, 6.184208118864554e-05, -0.0026979359424657012, -0.0002179584163330972, 0.00018427477667071201, 0.00046194266902999104, -0.00043012100846628876, -0.0001282813300972352, -0.00024649193246251144, 0.00021856020713362234, -0.0008925322174963002, 0.00011390192067406435, 0.0023872537134125293, -0.003470098675647293, 3.241730221992256e-05, -0.0017963179532922702, -0.006139630293307441, 0.003085026530005308, -0.001424807938971706, -0.001274028261726512, -0.0018217392320472037, -0.005396116816206553, -0.00032892710812215675, -0.0009527060426186562, 0.003638287347615954, 0.000213585274617345, -0.00010407084528318844, -0.0006758461564543294, -0.0023440348769743213, 7.332287835304374e-05, -0.0004188157072628762, -0.0015919436634266315, -0.002535330876158426, 0.00027103951193463427, -0.002204454393019306, -0.008199550969738345, -0.0036882782290835747, 0.0005043987416630813, -0.0009761326297701434, 0.0007512268890899065, 0.00997751998402241, 0.0024867199195313106, 0.000437479457638279, -0.001351260980326785, 0.008681811755647306, 0.008409111487725163, 0.0017603403052113276, 0.0013167984581317072, 0.0006714846559018328, 0.007958946969843783, -0.0021571952780214954, 0.004632584767417902, 0.00031731004095458815, 0.005205843252965248, -0.0037593119244861972, 0.010998499396588146, 0.004728987157864961, 0.00047344109520675026, 0.0037388963760759397, 0.004290932969103323, -0.00958139451116713, -0.0012605142917067047, -0.017917736000412687, 0.001213161285349984, -0.0019478077512052883, -0.0014631587899428533, -0.0060666305181715785, -0.0031629868159533947, -0.004045106780166674, -0.004237762521450805, 0.0003988989360991235, 0.005691454056938494, 0.0005050886862884019, -0.000708452050795727, -0.0009383795960050678, -0.003167730120980483, -0.005643685106171081, -0.009171104261583074, 0.00028941538408955303, -0.0023776515430778966, -0.01141459397356892, -0.002872204496502717, -0.008422712703008206, 0.0014651192349970366, 0.0009567026072596629, -0.002296218528907601, -0.00027923621707171, 0.0001568656582635608, 0.0007675267798219476, -0.003893125694685473, -0.001328149497989095, -0.00041188227402444433, -0.0031242024546712654, -0.008185731558088993, -0.0012589978281094538, 0.00035732447429082104, -0.003593197294032069, -0.0009126513004171777, 0.0005689793082794825, -6.453347455848135e-05, -0.000370580456240055, 0.00027525577572251603, -3.6502287718453985e-05, -0.0023074552362495337, -0.008800408964748985, -0.003197130631564349, -0.012735567198833397, -0.011369555764310424, -0.009968342103654826, -0.0025554221671136856, -0.00583015096095818, 0.005205725477384182, -0.005765322696708819, -0.001187471057840006, -0.0038131170425538263, -0.005269059242070773, -0.001457901181809218, -0.0008872733260566479, -0.00040371193116239823, -0.002953209587786554, -0.0037572196078657916, 0.00017578242169358216, 0.012856226517308778, -0.0007585457328395107, 0.00023855784057880136, -0.002848557009832109, -0.0078143168189716, -0.0005993502992025501, -0.008850219267620196, -0.023687601041420203, -0.005610423551353105, -0.0003403299695906581, -0.006706858874932609, -0.007827503516885104, -0.014051136392091657, -0.022939217235600662, -0.02596011328653758, -0.003562654711130153, 0.003463129745354787, -0.028571856551404706, -0.00550059567023331, -0.00032356541705980124, -0.028238911446790027, -0.007968311802305506, -0.01037121589383283, 0.0004606693922267757, -0.002119639438283083, 0.0003516650248051034, -0.005591090321129477, -0.024419747493691024, 0.0032479863456902347, -0.005691213937009998, -0.00693142530305102, -0.0015943101459882092, -0.005235125844634703, 0.0008105555605669723, -0.0125620482978267, 0.0002832520572871866, 0.001132864096134445, 2.4724404741327533e-05, -0.01047139853867296, 0.0009248645315039772, -0.00717563298583046, -0.004039134245337639, -0.0001474577104055852, 0.005452434480899826, -0.0006313927910276407, -0.0003103067794382186, -0.0003769289820586552, -0.004561481558340796, -0.0036435729461331556, -0.009657924645436577, -0.002214256542681148, -0.0010674573098212736, -0.007697048425683306, 0.0004783212144975346, -0.008294902511270963, 0.0014503858730579022, -0.0006695527928287648, -0.0007481991524662668, -0.002096718651721452, -0.00042882442384239994, 0.0009084657766876836, -0.0014935570421196198, 0.0002254736416018588, -0.005238157092741447, -0.008079029222606837, -0.006891672579447783, -0.0002629402254594737, 0.00025512603336930173, -0.0019430058872137446, -0.0008653866453605108, -0.00024594540031064316, -5.6779565724363046e-05, -0.0003578553352771019, 0.00012550817031120598, -0.00025997042736281006, -0.001703867085843691, -0.005334834665414068, -0.00301589764323118, -0.004383940234229944, -0.00480797277897238, -0.003575541296818696, -0.002667321393234112, -0.007993539439891006, 0.003251328743656745, -0.0027013261514131637, -0.00016208073325996164, -0.004932725360870971, -0.0044034820373297645, 0.00040565324724939344, -0.0013124895211601428, -0.0030436746488495814, 0.0005626703200726224, -0.002225319804398107, -0.0008710457613324563, 0.0037522977243465983, -0.0003468641212648338, -2.4294197935925688e-05, -0.0026336931910834194, -0.004833058857956041, -0.0005984916645737082, -0.0060355944274236295, -0.01286660511385851, -0.0038610615374091717, 0.001995509530078654, -0.0032224721877091023, -0.006130294682895054, -0.0037421111298875376, -0.012496396898601825, -0.011586315457566054, -0.0036555180443077484, 0.009562796169411247, -0.015703028013520193, 0.000995143698275358, 0.0023073567996806377, -0.012070600490433292, -0.008180728910568091, -0.0003341755914415545, 0.0022667551602119085, 0.003565200689467418, -0.0008869286363432094, -0.004930951287559382, -0.00936665590930541, 0.0031606152021559364, 0.00018437759361600303, 0.0008964489297366152, 0.005620334525837989, -0.003769199471557072, 0.00037360292681309915, -0.009912639499390807, 0.0013776112486942882, 0.0014945162178718885, 0.000638496173621661, -0.004502567174247463, -0.0008574358280195593, -0.004287831942328735, -0.0008279281270960892, 0.0001551115285627347, 0.003339987677618561, 0.0006796780749992459, 0.0005205564413375274, 0.0006673594565830276, -0.0017082512342949663, 0.0003114885205619533, -0.004551676888916889, -0.0010910595961164393, 0.0006831180676498683, -0.0016408355553787501, 0.0008798239318028589, -0.004879380610297357, -0.0021184385040164955, -0.0011031425514225202, 0.000599749314037723, -0.0018960030931329808, -0.0004234857646296941, -0.0010245342138381908, 0.0054267178091419455, 0.0008345615809522977, 0.0010613250117759154, -0.006168406670414073, -0.005988220481718098, 0.000767707365118997, -0.0009078767506419672, -0.0013669789623311779, -0.00022315213188939254, -0.0004147290626814737, -2.2010516496084875e-05, -0.0006187845153059479, 5.11100207157833e-05, 0.00018115652804162607, -0.0012034245203647043, -0.0033117197682566157, -0.000758293489238171, -0.00170611679174415, 0.001323155104417273, 0.0003704052834895277, -0.003045200354908628, -0.00017451612841834496, 0.0015851242605640081, -0.0007412108817819174, -0.001605051675545921, -0.0036614036764068825, -0.0006224309196298794, 0.00026434453160073534, -0.00022002339577420482, 0.0019322216475592402, 0.0015347630256304923, -0.0024941446844604087, 5.6123092818327613e-05, 0.005137438396225935, 0.0005050789493994401, -0.00032902116016619853, -0.0019195257314224001, -0.0021925245123866066, -0.00012312329654018406, -0.0038368318834237636, -0.0022309960903037496, -0.00569826032736942, 0.0010854238790678483, 0.0013894589674022795, -0.004553111713877067, -0.0023364407602208814, 0.012254348983420371, 0.009730473860487976, -0.003611174983310095, 0.0016250340069640101, 0.009428144851865605, -0.003948528145741648, -0.00019324480385329856, 0.007382767487241981, -0.0014959288322273778, -0.011821994391204793, 0.0020350675869791335, -0.011340537226294911, 0.0009166502477956362, -0.0024119762943364265, 0.012845168647646399, 0.0064490032731310265, 0.00031004470163153997, 0.006110147396582111, 0.007251366966245842]
    R_minus = pd.DataFrame(np.array(R2_omit).reshape(-1, 94)*100, index=models, columns=CHARAS_LIST).T
    char_ranks = R_minus.T.sum().argsort().argsort().index.to_list()
    char_ranks.reverse()
    
    plt.figure(figsize=(8, 15), dpi=200)
    sns.heatmap(R_minus.T[char_ranks].T, cmap='Blues', linewidths=0.6)
    plt.savefig('imgs/omit_char_R2_bias.png', bbox_inches='tight')
    plt.close()
    