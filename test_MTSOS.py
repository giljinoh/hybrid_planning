#include "MTSOS.h"
import MTSOS
import MTSOS_header
import math
import time
import matplotlib.pyplot as plt


# from drawnow import *

def plot_arrow(x, y, yaw, length=1.0, width=0.5, fc="r", ec="k"):  # pragma: no cover
    """
    Plot arrow
    """

    if not isinstance(x, float):
        for (ix, iy, iyaw) in zip(x, y, yaw):
            plot_arrow(ix, iy, iyaw)
    else:
        plt.arrow(x, y, length * math.cos(yaw), length * math.sin(yaw),
                  fc=fc, ec=ec, head_width=width, head_length=width)
        plt.plot(x, y)

def main(arr):
    p_params = MTSOS_header.problem_params()
    a_flags = MTSOS_header.algorithm_flags()
    a_params = MTSOS_header.algorithm_params()
    o_params = MTSOS_header.optional_params()
    b = []
    u = []
    v = []
    timers = []
    fx = 0
    iterations = 0
    i = 0
    total_time = 0
    vars = [1,1]
    S = []
    S_S =  arr[::-1] #[10.0, 10.0, 10.379662760416668, 12.576838975694447, 10.768552083333333, 15.058461805555554, 11.166519531250003, 17.446527343750002, 11.573416666666665, 19.742694444444442, 11.989095052083332, 21.948621961805557, 12.413406250000003, 24.065968750000007, 12.846201822916669, 26.096393663194448, 13.287333333333335, 28.04155555555556, 13.736652343749999, 29.903113281249997, 14.194010416666668, 31.682725694444443, 14.659259114583337, 33.38205164930556, 15.13225, 35.00275000000001, 15.612834635416668, 36.54647960069444, 16.100864583333333, 38.01489930555556, 16.59619140625, 39.40966796875, 17.09866666666667, 40.73244444444445, 17.608141927083338, 41.98488758680557, 18.124468750000002, 43.16865625, 18.647498697916664, 44.285409288194444, 19.177083333333332, 45.33680555555555, 19.71307421875, 46.32450390625, 20.255322916666668, 47.25016319444444, 20.80368098958333, 48.11544227430556, 21.358, 48.922000000000004, 21.918131510416668, 49.67149522569445, 22.483927083333338, 50.365586805555566, 23.055238281250006, 51.005933593750015, 23.631916666666665, 51.59419444444443, 24.213813802083337, 52.132028211805554, 24.80078125, 52.62109375, 25.392670572916664, 53.06304991319444, 25.989333333333335, 53.45955555555555, 26.590621093750002, 53.812269531249996, 27.196385416666672, 54.12285069444446, 27.806477864583336, 54.39295789930556, 28.420750000000005, 54.62425, 29.039053385416675, 54.818385850694455, 29.661239583333334, 54.97702430555555, 30.287160156250003, 55.10182421875, 30.916666666666664, 55.19444444444444, 31.54961067708334, 55.25654383680555, 32.185843750000004, 55.28978125000001, 32.82521744791667, 55.29581553819445, 33.46758333333334, 55.27630555555556, 34.11279296875, 55.23291015625, 34.76069791666667, 55.167288194444446, 35.41114973958334, 55.08109852430556, 36.06400000000001, 54.97600000000001, 36.719100260416674, 54.85365147569445, 37.376302083333336, 54.71571180555556, 38.035457031250004, 54.563839843749996, 38.696416666666664, 54.39969444444444, 39.35903255208334, 54.22493446180555, 40.02315625000001, 54.04121875, 40.688639322916664, 53.85020616319444, 41.355333333333334, 53.65355555555555, 42.02308984375, 53.452925781249995, 42.691760416666675, 53.24997569444445, 43.36119661458334, 53.04636414930555, 44.03125, 52.84375, 44.701772135416675, 52.64379210069445, 45.37261458333333, 52.44814930555555, 46.04362890625001, 52.258480468749994, 46.714666666666666, 52.07644444444444, 47.38557942708333, 51.90370008680555, 48.056218750000006, 51.74190625, 48.72643619791668, 51.592721788194446, 49.39608333333334, 51.45780555555555, 50.06501171875, 51.33881640625, 50.73307291666667, 51.23741319444444, 51.400118489583335, 51.15525477430555, 52.06600000000001, 51.09400000000001, 52.73056901041667, 51.05530772569443, 53.39367708333333, 51.04083680555555, 54.05517578125, 51.05224609375, 54.714916666666674, 51.091194444444454, 55.372751302083344, 51.15934071180556, 56.02853125, 51.258343749999995, 56.68210807291666, 51.389862413194436, 57.333333333333336, 51.55555555555556, 57.98193055555555, 51.75657638888888, 58.62711111111111, 51.99205555555555, 59.26795833333335, 52.260618055555554, 59.90355555555557, 52.5608888888889, 60.532986111111114, 52.89149305555556, 61.15533333333334, 53.25105555555557, 61.76968055555556, 53.638201388888895, 62.375111111111124, 54.05155555555555, 62.970708333333334, 54.48974305555556, 63.55555555555555, 54.95138888888888, 64.12873611111111, 55.435118055555556, 64.68933333333334, 55.939555555555565, 65.23643055555557, 56.4633263888889, 65.76911111111113, 57.00505555555556, 66.28645833333334, 57.563368055555564, 66.78755555555554, 58.13688888888889, 67.27148611111112, 58.724243055555554, 67.73733333333334, 59.32405555555556, 68.18418055555556, 59.93495138888889, 68.6111111111111, 60.55555555555555, 69.01720833333334, 61.18449305555557, 69.40155555555557, 61.82038888888891, 69.76323611111111, 62.461868055555556, 70.10133333333334, 63.107555555555564, 70.41493055555557, 63.7560763888889, 70.70311111111113, 64.40605555555557, 70.96495833333333, 65.05611805555556, 71.19955555555556, 65.7048888888889, 71.4059861111111, 66.35099305555556, 71.58333333333333, 66.99305555555556, 71.73068055555557, 67.6297013888889, 71.84711111111113, 68.25955555555556, 71.93170833333333, 68.88124305555557, 71.98355555555557, 69.4933888888889, 72.0017361111111, 70.09461805555556, 71.98533333333333, 70.68355555555556, 71.93343055555556, 71.2588263888889, 71.84511111111111, 71.81905555555555, 71.71945833333332, 72.36286805555554, 71.55555555555556, 72.88888888888889, 71.35288975694444, 73.39590798611111, 71.11256250000001, 73.88337500000002, 70.83607899305555, 74.35090451388889, 70.52494444444444, 74.79811111111111, 70.1806640625, 75.224609375, 69.80474305555555, 75.6300138888889, 69.39868663194444, 76.01393923611113, 68.964, 76.376, 68.50218836805556, 76.7158107638889, 68.01475694444446, 77.03298611111111, 67.5032109375, 77.327140625, 66.96905555555556, 77.5978888888889, 66.41379600694444, 77.84484548611111, 65.83893749999999, 78.06762499999998, 65.24598524305554, 78.26584201388889, 64.63644444444444, 78.4391111111111, 64.0118203125, 78.58704687500001, 63.37361805555556, 78.7092638888889, 62.72334288194445, 78.80537673611111, 62.06249999999999, 78.875, 61.39259461805555, 78.9177482638889, 60.71513194444444, 78.93323611111111, 60.03161718749999, 78.921078125, 59.34355555555554, 78.88088888888889, 58.65245225694444, 78.81228298611111, 57.95981249999999, 78.714875, 57.26714149305555, 78.58827951388889, 56.57594444444445, 78.43211111111113, 55.88772656249999, 78.24598437499998, 55.20399305555556, 78.02951388888889, 54.52624913194444, 77.7823142361111, 53.855999999999995, 77.50399999999999, 53.19475086805555, 77.19418576388888, 52.54400694444444, 76.85248611111112, 51.9052734375, 76.478515625, 51.28005555555555, 76.07188888888888, 50.66985850694444, 75.6322204861111, 50.07618749999999, 75.159125, 49.500547743055535, 74.65221701388886, 48.94444444444444, 74.1111111111111, 48.4093828125, 73.53542187499998, 47.89686805555556, 72.9247638888889, 47.40840538194444, 72.2787517361111, 46.945499999999996, 71.597, 46.50965711805556, 70.8791232638889, 46.10238194444444, 70.12473611111109, 45.7251796875, 69.33345312500002, 45.379555555555555, 68.50488888888889, 45.06701475694444, 67.63865798611108, 44.7890625, 66.734375, 44.54720399305556, 65.79165451388889, 44.34294444444444, 64.81011111111111, 44.1777890625, 63.789359375, 44.053243055555555, 62.72901388888886, 43.97081163194444, 61.628689236111114, 43.932, 60.48799999999998, 43.93831336805556, 59.30656076388891, 43.991256944444444, 58.0839861111111, 44.092335937499996, 56.819890624999964, 44.24305555555556, 55.513888888888886, 44.44492100694444, 54.16559548611109, 44.6994375, 52.77462500000001, 45.00811024305556, 51.34059201388888, 45.372444444444454, 49.863111111111074, 45.7939453125, 48.341796875, 46.27411805555556, 46.77626388888886, 46.81446788194444, 45.166126736111124, 47.4165, 43.51099999999998, 48.08171961805557, 41.81049826388885, 48.81163194444444, 40.064236111111114, 49.60774218750001, 38.27182812499997, 50.47155555555558, 36.43288888888884, 51.40457725694445, 34.54703298611109, 52.40831250000003, 32.61387499999996, 53.48426649305556, 30.63302951388889, 54.63394444444446, 28.60411111111108, 55.858851562500035, 26.52673437499994, 57.16049305555557, 24.400513888888874, 58.54037413194447, 22.225064236111063, 60.0, 20.0]
    for i in range(len(S_S)):
        S.append(S_S[i]/100.0)
    
    print(S)
    the = []
    # CS = [0.000000,  0.001248,  0.002954,  0.004983,  0.007394,  0.010210,  0.013643,  0.014441,  0.014642,  0.014987,  0.015282,  0.015436,  0.015921,  0.016379,  0.016615,  0.016874,  0.017019,  0.017939,  0.018209,  0.017147,  0.016255,  0.015614,  0.015000,  0.014999,  0.015140,  0.015777,  0.015837,  0.012775,  0.012825,  0.013929,  0.014409,  0.014306,  0.014674,  0.015361,  0.015701,  0.016045,  0.016590,  0.017262,  0.017708,  0.018387,  0.019205,  0.019875,  0.020919,  0.021653,  0.022742,  0.024019,  0.025191,  0.026628,  0.028121,  0.029744,  0.031642,  0.033880,  0.035130,  0.034216,  0.033045,  0.031734,  0.030320,  0.029327,  0.028665,  0.027601,  0.026790,  0.025887,  0.024961,  0.024437,  0.023603,  0.022730,  0.022278,  0.021683,  0.021215,  0.021092,  0.020321,  0.019131,  0.018043,  0.018350,  0.017703,  0.013709,  0.015169,  0.012000,  0.011085,  0.010632,  0.009942,  0.009587,  0.009464,  0.009383,  0.009522,  0.009683,  0.009880,  0.010104,  0.010380,  0.010796,  0.011121,  0.011435,  0.012001,  0.012495,  0.012996,  0.013658,  0.014309,  0.015028,  0.015871,  0.016878,  0.017838,  0.018994,  0.020312,  0.021640,  0.023427,  0.022623,  0.021481,  0.020606,  0.019838,  0.019001,  0.018364,  0.017747,  0.017041,  0.016446,  0.015935,  0.015538,  0.015160,  0.014782,  0.014450,  0.014329,  0.014388,  0.014678,  0.015093,  0.015327,  0.016016,  0.014344,  0.012237,  0.008855,  0.008777,  0.009209,  0.009250,  0.009349,  0.009474,  0.009855,  0.010197,  0.010339,  0.010653,  0.011100,  0.011360,  0.011704,  0.012176,  0.012553,  0.013021,  0.013465,  0.013981,  0.014575,  0.015113,  0.015722,  0.016424,  0.017117,  0.017955,  0.018719,  0.019680,  0.020786,  0.021826,  0.023089,  0.024495,  0.025975,  0.027584,  0.029458,  0.031448,  0.031701,  0.030492,  0.029455,  0.028496,  0.027523,  0.026570,  0.025621,  0.024742,  0.024166,  0.023423,  0.022494,  0.021825,  0.021424,  0.020812,  0.020158,  0.019634,  0.019011,  0.018597,  0.017973,  0.017427,  0.017021,  0.017660,  0.016982,  0.015893,  0.015283,  0.015488,  0.014263,  0.013332,  0.012345,  0.011929,  0.012355,  0.008896,  0.009019,  0.008730,  0.008369,  0.008117,  0.007836,  0.007583,  0.007318,  0.007739] 
    # CS =  [0.000000,  0.001974,  0.004209,  0.006695,  0.009462,  0.012540,  0.015871,  0.019249,  0.022690,  0.026210,  0.029826,  0.033552,  0.037392,  0.040222,  0.041035,  0.041863,  0.042712,  0.043586,  0.044490,  0.045426,  0.046398,  0.047406,  0.048456,  0.049547,  0.050685,  0.051870,  0.053108,  0.054399,  0.055749,  0.057161,  0.058638,  0.060186,  0.061809,  0.063512,  0.065301,  0.067185,  0.069158,  0.071219,  0.073394,  0.075686,  0.078105,  0.080657,  0.083353,  0.086201,  0.089213,  0.092398,  0.095768,  0.099334,  0.103107,  0.107097,  0.111314,  0.115766,  0.120434,  0.125028,  0.126516,  0.122335,  0.115779,  0.108583,  0.101515,  0.095106,  0.089439,  0.084391,  0.079863,  0.075780,  0.072084,  0.068725,  0.065662,  0.062863,  0.060300,  0.057950,  0.055795,  0.053824,  0.052038,  0.050493,  0.052983,  0.046861,  0.045718,  0.044568,  0.043498,  0.042504,  0.041621,  0.041135,  0.041165,  0.041569,  0.042253,  0.043172,  0.044304,  0.045638,  0.047173,  0.048914,  0.050873,  0.053066,  0.055514,  0.058247,  0.061298,  0.064711,  0.068542,  0.072865,  0.077730,  0.082817,  0.087492,  0.090499,  0.089368,  0.086918,  0.084567,  0.082293,  0.080089,  0.077957,  0.075899,  0.073917,  0.072011,  0.070181,  0.068425,  0.066745,  0.065141,  0.063616,  0.062176,  0.060838,  0.059638,  0.058722,  0.058592,  0.059122,  0.060458,  0.061752,  0.063031,  0.064323,  0.065630,  0.066938,  0.068366,  0.066333,  0.063109,  0.060303,  0.057874,  0.055766,  0.053932,  0.052334,  0.050942,  0.049731,  0.048686,  0.047794,  0.047062,  0.046566,  0.051591,  0.044737,  0.044571,  0.044354,  0.044196,  0.044116,  0.044122,  0.044218,  0.044409,  0.044700,  0.045098,  0.045611,  0.046250,  0.047026,  0.047957,  0.049060,  0.050361,  0.051886,  0.053673,  0.055463,  0.056533,  0.056952,  0.056837,  0.056331,  0.055577,  0.054686,  0.053730,  0.052750,  0.051771,  0.050803,  0.049854,  0.048928,  0.048025,  0.047148,  0.046295,  0.045468,  0.044664,  0.043885,  0.043128,  0.042393,  0.041680,  0.040988,  0.040315,  0.039662,  0.039027,  0.038409,  0.037809,  0.037225,  0.036656,  0.036103,  0.035563,  0.035037,  0.034523,  0.034020,  0.033526,  0.033039,  0.032550,  0.033224]
    # CS_t = 4.7221 #[0.000000, 0.044428 , 0.109302 , 0.146700 , 0.179101 , 0.209258 , 0.237964 , 0.264721 , 0.289372 , 0.312528 , 0.334599 , 0.355874 , 0.376542 , 0.393926 , 0.403125 , 0.407176 , 0.411274 , 0.415442 , 0.419700 , 0.424061 , 0.428535 , 0.433131 , 0.437857 , 0.442719 , 0.447726 , 0.452884 , 0.458202 , 0.463687 , 0.469349 , 0.475195 , 0.481237 , 0.487482 , 0.493943 , 0.500630 , 0.507556 , 0.514742 , 0.522181 , 0.529849 , 0.537782 , 0.546024 , 0.554584 , 0.563474 , 0.572710 , 0.582308 , 0.592285 , 0.602655 , 0.613434 , 0.624637 , 0.636275 , 0.648358 , 0.660893 , 0.673881 , 0.687280 , 0.700629 , 0.709283 , 0.705454 , 0.690026 , 0.669782 , 0.648134 , 0.627008 , 0.607457 , 0.589565 , 0.573100 , 0.557882 , 0.543767 , 0.530639 , 0.518400 , 0.506972 , 0.496286 , 0.486288 , 0.476938 , 0.468211 , 0.460118 , 0.452823 , 0.454885 , 0.446653 , 0.430291 , 0.424928 , 0.419673 , 0.414728 , 0.410178 , 0.406830 , 0.405711 , 0.406778 , 0.409440 , 0.413333 , 0.418263 , 0.424115 , 0.430823 , 0.438358 , 0.446716 , 0.455910 , 0.465974 , 0.476958 , 0.488927 , 0.501968 , 0.516188 , 0.531741 , 0.548737 , 0.566581 , 0.583571 , 0.596621 , 0.599774 , 0.593763 , 0.585624 , 0.577672 , 0.569866 , 0.562206 , 0.554705 , 0.547375 , 0.540226 , 0.533265 , 0.526499 , 0.519933 , 0.513578 , 0.507448 , 0.501572 , 0.496004 , 0.490862 , 0.486535 , 0.484384 , 0.485208 , 0.489031 , 0.494380 , 0.499558 , 0.504679 , 0.509803 , 0.514907 , 0.520193 , 0.519020 , 0.508766 , 0.496781 , 0.486136 , 0.476719 , 0.468382 , 0.460999 , 0.454469 , 0.448708 , 0.443654 , 0.439267 , 0.435557 , 0.432730 , 0.442929 , 0.438648 , 0.422630 , 0.421724 , 0.420833 , 0.420266 , 0.420089 , 0.420332 , 0.421014 , 0.422158 , 0.423787 , 0.425931 , 0.428625 , 0.431913 , 0.435846 , 0.440486 , 0.445907 , 0.452197 , 0.459461 , 0.467181 , 0.473273 , 0.476414 , 0.477052 , 0.475747 , 0.473089 , 0.469598 , 0.465647 , 0.461472 , 0.457206 , 0.452927 , 0.448676 , 0.444476 , 0.440343 , 0.436282 , 0.432299 , 0.428395 , 0.424571 , 0.420826 , 0.417159 , 0.413569 , 0.410053 , 0.406611 , 0.403240 , 0.399938 , 0.396703 , 0.393534 , 0.390428 , 0.387382 , 0.384395 , 0.381465 , 0.378588 , 0.375762 , 0.372983 , 0.370247 , 0.367545 , 0.364867 , 0.362180 , 0.362690]
    b = None
    v = None
    u = None
    timers = None
    fx = 0
    iterations = 0
    p_params.S = S
    p_params.S_length = 100
    # p_params.S_length = 201
    p_params.initial_velocity = 0
    p_params.State_size = 2
    p_params.U_size = 2
    a_flags.timer = 1
    a_flags.display = 0
    a_flags.kappa = 1
    a_params.kappa = 0
    a_params.alpha = 0
    a_params.beta = 0
    a_params.epsilon = .01
    a_params.MAX_ITER = 0
    o_params.variables = vars
    o_params.variables_length = 2
    o_params.initial_b = None #0
    o_params.initial_u = None #0
    pi = 3.1415926535
    print("Running test for MTSOS.\n")
    the = []
    d = []
    for i in range (p_params.S_length-1):
        theta = math.atan2(S[p_params.State_size*i+2] - S[p_params.State_size*i],S[p_params.State_size*i+3] - S[p_params.State_size*i+1]) # absolute angle
        d.append(((S[p_params.State_size*i+3] - S[p_params.State_size*i+1])**2 + (S[p_params.State_size*i+2] - S[p_params.State_size*i])**2)**0.5)
        # theta = 180/pi*theta
        the.append(theta)
    start = time.time()
    status, fx, b = MTSOS.so_MTSOS(p_params, a_flags, a_params, o_params, b, u, v, fx, timers, iterations)
    #print(fx, d)
    for i in range(p_params.S_length-1):
        b[i] = math.sqrt(b[i])
    t = []
    summ = 0
    for i in range(p_params.S_length-1):
        temp = d[i] / b[i+1]
        summ += temp
        t.append(temp)
    print(t, summ)
    # t = fx / p_params.S_length * 0.58
    # CS_t = CS_t/p_params.S_length

        # CS[i] = math.sqrt(CS[i])
            
        
    print(b)
    print(the)
    dis=[]
    dis_c = []
    for i in range(p_params.S_length-1):
        dis.append(t[i]*b[i+1])
        #dis.append(d[i])
        # dis.append(t * b[i])
        # dis_c.append(CS_t*CS[i])
        # dis.append(2/p_params.S_length)
        # dis_c.append(2/p_params.S_length)
    print(dis)
    total_time = time.time() - start
    print("the optimal time to traverse is ", fx)

    # ax = plt.subplots()
    # ax.set_xlim(0,1)
    # ax.set_ylim(0,1)

    X = [S[0]]
    Y = [S[1]]
    # X_c = [S[0]]
    # Y_c = [S[1]]
    for i in range(1,p_params.S_length-1):
        X.append(math.cos(the[i-1]) * dis[i-1] + X[i-1])
        Y.append(math.sin(the[i-1]) * dis[i-1] + Y[i-1])
        # X_c.append(math.cos(the[i-1]) * dis_c[i-1] + X_c[i-1])
        # Y_c.append(math.sin(the[i-1]) * dis_c[i-1] + Y_c[i-1])
    err = 0
    temp = 0

    # if show_animation:  # pragma: no cover
    for i, _ in enumerate(b):
        plt.cla()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event: [exit(0) if event.key == 'escape' else None])
        plt.grid(True)
        plt.axis("equal")
        plot_arrow(S[1]*100,S[0]*100,  the[0])
        plot_arrow( S[-1]*100,S[-2]*100, the[-1])
        plot_arrow( S[p_params.State_size*(i+1)+1]*100,S[p_params.State_size*(i+1)]*100, the[i+1])
        # plt.title("Time[s]:" + str(time[i])[0:4] +
        #           " v[m/s]:" + str(rv[i])[0:4] +
        #           " a[m/ss]:" + str(ra[i])[0:4] +
        #           " jerk[m/sss]:" + str(rj[i])[0:4],
        #           )
        plt.pause(0.001)
    # for i in range(len(X)):
    #     err += math.sqrt((X_c[i]-X[i])**2 + (Y_c[i]-Y[i])**2)
    #     temp = max(temp, math.sqrt((X_c[i]-X[i])**2 + (Y_c[i]-Y[i])**2))
    plt.plot(Y, X, '-r',label='python')
    # plt.plot(X_c, Y_c, '-g', label="C ")
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.title('total error '+str(err))
    
    print("max", temp)

    plt.plot( S[1::2],S[0::2], '-b', label="fixed path")
    plt.legend(loc='upper right')
    # line, = plt.plot([],[], 'bo')
    # def update(frame):
        
    #     # X.append(S[frame])
    #     # Y.append(S[frame+1])
    #     X.append(math.cos(the[frame]) * dis[frame])
    #     Y.append(math.sin(the[frame]) * dis[frame])
    #     line.set_data(X,Y)
    #     return line,

    # ani = FuncAnimation(fig,update,frames=range(0,p_params.S_length,2), interval=50)
    plt.show()

    # for i in S:
    #     if i % 2 ==0:
    #         X = np.apend(X,i)
    #     else:
    #         Y = np.append(Y,i)
    #     drawnow(show_plot)


    if abs(fx) <10: #1e-3:

        #for i in range(9):
            #total_time += timers[i]
        print("the time to compute is",total_time,"seconds.\n")
        print("MTSOS is working!\n")
	
    else:
	    print("The optimal time to traverse should have been 8.3287. \n There appears to be a problem with your BLAS linking.\n  See http://www.stanford.edu/~boyd/MTSOS/install.html for more information.\n")
        

    if b != None:
        b.clear()
    if u != None:
        u.clear()
    if v != None:
        v.clear()
    if timers != None:
        timers.clear()

    
    return 0


if __name__ == "__main__":
    main()