import numpy as np
import torch
import torch.nn.functional as F

zz = np.array([[-19.70066725,  11.49479733,  -0.57532162],
       [-19.70160604,  12.49479688,  -0.61815465],
       [-19.70254482,  13.49479644,  -0.66098768],
       [-19.7034836 ,  14.494796  ,  -0.7038207 ],
       [-19.70394189,  14.98296258,  -0.72473036]])

z = np.array([-19.70066725, 11.49479733, -0.57532162])

aa = np.array([[3.50520267e+00, 4.97006673e+01],
       [1.76112005e+00, 4.97023046e+01],
       [1.70374229e-02, 4.97039419e+01]])

patch = [30, 60]
# vv = np.array(patch)/2 - zz[:, :2][:, ::-1]

bb = (np.array([21.31085124, 45.31261538]), np.array([21.63881888, 54.0842686 ]))
points =[]
for point in bb:
       new_point = point[::-1]
       new_point[:1] = -new_point[:1]
       new_point += np.array([patch[1], -patch[0]])/2
       points.append(new_point)
# print(points)

aa[:, :] = -aa[:,:]
# print(aa)



# 示例数据
input = torch.tensor([[[0.5107, 0.5244],
         [0.5125, 0.5240],
         [0.5141, 0.5231],
         [0.5158, 0.5221],
         [0.5172, 0.5207],
         [0.5186, 0.5191],
         [0.5198, 0.5174],
         [0.5208, 0.5154],
         [0.5217, 0.5135],
         [0.5226, 0.5115],
         [0.5233, 0.5096],
         [0.5237, 0.5076],
         [0.5241, 0.5060],
         [0.5241, 0.5042],
         [0.5240, 0.5028],
         [0.5238, 0.5017],
         [0.5232, 0.5007],
         [0.5225, 0.5011],
         [0.5213, 0.5033],
         [0.5192, 0.5048],
         [0.5171, 0.5065],
         [0.5144, 0.5076],
         [0.5118, 0.5086],
         [0.5091, 0.5093],
         [0.5064, 0.5095],
         [0.5039, 0.5093],
         [0.5019, 0.5090],
         [0.5001, 0.5081],
         [0.4989, 0.5068],
         [0.4985, 0.5053],
         [0.4987, 0.5031],
         [0.4998, 0.5006],
         [0.5018, 0.4974],
         [0.5049, 0.4937],
         [0.5040, 0.4938],
         [0.5017, 0.4940],
         [0.4997, 0.4936],
         [0.4977, 0.4925],
         [0.4959, 0.4909],
         [0.4939, 0.4887],
         [0.4921, 0.4863],
         [0.4904, 0.4839],
         [0.4888, 0.4815],
         [0.4873, 0.4793],
         [0.4858, 0.4773],
         [0.4842, 0.4757],
         [0.4825, 0.4745],
         [0.4812, 0.4743],
         [0.4795, 0.4746],
         [0.4780, 0.4760],
         [0.4768, 0.4780],
         [0.4814, 0.4812],
         [0.4854, 0.4844],
         [0.4891, 0.4876],
         [0.4922, 0.4905],
         [0.4949, 0.4933],
         [0.4974, 0.4959],
         [0.4994, 0.4980],
         [0.5012, 0.4998],
         [0.5029, 0.5012],
         [0.5044, 0.5021],
         [0.5057, 0.5023],
         [0.5071, 0.5021],
         [0.5083, 0.5009],
         [0.5096, 0.4991],
         [0.5110, 0.4965],
         [0.5123, 0.4928],
         [0.5119, 0.4898],
         [0.5087, 0.4872],
         [0.5055, 0.4841],
         [0.5029, 0.4812],
         [0.5005, 0.4780],
         [0.4986, 0.4749],
         [0.4971, 0.4718],
         [0.4959, 0.4687],
         [0.4953, 0.4657],
         [0.4953, 0.4631],
         [0.4958, 0.4606],
         [0.4967, 0.4583],
         [0.4985, 0.4566],
         [0.5008, 0.4551],
         [0.5037, 0.4540],
         [0.5073, 0.4534],
         [0.5115, 0.4532],
         [0.5085, 0.4516],
         [0.5036, 0.4498],
         [0.4998, 0.4492],
         [0.4968, 0.4494],
         [0.4948, 0.4505],
         [0.4933, 0.4523],
         [0.4926, 0.4548],
         [0.4925, 0.4581],
         [0.4928, 0.4619],
         [0.4937, 0.4664],
         [0.4948, 0.4713],
         [0.4960, 0.4764],
         [0.4973, 0.4819],
         [0.4989, 0.4879],
         [0.5001, 0.4938],
         [0.5015, 0.5001]],

        [[0.4958, 0.5049],
         [0.5030, 0.5075],
         [0.5090, 0.5092],
         [0.5142, 0.5103],
         [0.5184, 0.5107],
         [0.5218, 0.5107],
         [0.5245, 0.5104],
         [0.5263, 0.5097],
         [0.5277, 0.5091],
         [0.5286, 0.5086],
         [0.5291, 0.5082],
         [0.5292, 0.5080],
         [0.5292, 0.5085],
         [0.5287, 0.5092],
         [0.5283, 0.5108],
         [0.5279, 0.5133],
         [0.5274, 0.5164],
         [0.5269, 0.5190],
         [0.5257, 0.5206],
         [0.5236, 0.5218],
         [0.5216, 0.5232],
         [0.5190, 0.5241],
         [0.5163, 0.5248],
         [0.5136, 0.5254],
         [0.5108, 0.5255],
         [0.5080, 0.5252],
         [0.5057, 0.5248],
         [0.5035, 0.5237],
         [0.5016, 0.5221],
         [0.5004, 0.5203],
         [0.4997, 0.5178],
         [0.4997, 0.5147],
         [0.5003, 0.5108],
         [0.5017, 0.5063],
         [0.5010, 0.5046],
         [0.4999, 0.5028],
         [0.4993, 0.5009],
         [0.4992, 0.4987],
         [0.4995, 0.4964],
         [0.4998, 0.4938],
         [0.5004, 0.4913],
         [0.5012, 0.4889],
         [0.5020, 0.4866],
         [0.5029, 0.4846],
         [0.5036, 0.4828],
         [0.5040, 0.4812],
         [0.5041, 0.4799],
         [0.5041, 0.4793],
         [0.5033, 0.4791],
         [0.5023, 0.4795],
         [0.5010, 0.4805],
         [0.5058, 0.4822],
         [0.5097, 0.4841],
         [0.5131, 0.4863],
         [0.5157, 0.4885],
         [0.5177, 0.4907],
         [0.5193, 0.4929],
         [0.5202, 0.4946],
         [0.5208, 0.4963],
         [0.5211, 0.4976],
         [0.5211, 0.4984],
         [0.5207, 0.4985],
         [0.5203, 0.4981],
         [0.5195, 0.4968],
         [0.5188, 0.4948],
         [0.5182, 0.4919],
         [0.5173, 0.4877],
         [0.5153, 0.4837],
         [0.5109, 0.4795],
         [0.5067, 0.4754],
         [0.5034, 0.4722],
         [0.5004, 0.4693],
         [0.4980, 0.4669],
         [0.4962, 0.4650],
         [0.4947, 0.4635],
         [0.4937, 0.4624],
         [0.4933, 0.4620],
         [0.4932, 0.4619],
         [0.4935, 0.4621],
         [0.4944, 0.4630],
         [0.4956, 0.4641],
         [0.4972, 0.4657],
         [0.4990, 0.4676],
         [0.5012, 0.4699],
         [0.4977, 0.4686],
         [0.4930, 0.4666],
         [0.4899, 0.4656],
         [0.4879, 0.4653],
         [0.4872, 0.4659],
         [0.4871, 0.4668],
         [0.4880, 0.4685],
         [0.4895, 0.4708],
         [0.4915, 0.4735],
         [0.4940, 0.4768],
         [0.4965, 0.4804],
         [0.4990, 0.4843],
         [0.5013, 0.4883],
         [0.5036, 0.4928],
         [0.5052, 0.4972],
         [0.5064, 0.5019]],

        [[0.5015, 0.5171],
         [0.5068, 0.5154],
         [0.5116, 0.5137],
         [0.5162, 0.5124],
         [0.5201, 0.5111],
         [0.5237, 0.5100],
         [0.5269, 0.5092],
         [0.5294, 0.5083],
         [0.5316, 0.5077],
         [0.5334, 0.5074],
         [0.5347, 0.5072],
         [0.5354, 0.5071],
         [0.5359, 0.5074],
         [0.5356, 0.5076],
         [0.5350, 0.5081],
         [0.5339, 0.5089],
         [0.5322, 0.5097],
         [0.5298, 0.5108],
         [0.5261, 0.5128],
         [0.5219, 0.5145],
         [0.5183, 0.5166],
         [0.5144, 0.5182],
         [0.5108, 0.5198],
         [0.5073, 0.5212],
         [0.5040, 0.5221],
         [0.5009, 0.5226],
         [0.4982, 0.5228],
         [0.4957, 0.5224],
         [0.4934, 0.5213],
         [0.4917, 0.5198],
         [0.4903, 0.5174],
         [0.4893, 0.5142],
         [0.4887, 0.5101],
         [0.4886, 0.5050],
         [0.4867, 0.5032],
         [0.4844, 0.5012],
         [0.4829, 0.4989],
         [0.4818, 0.4961],
         [0.4815, 0.4931],
         [0.4813, 0.4898],
         [0.4817, 0.4865],
         [0.4824, 0.4834],
         [0.4834, 0.4805],
         [0.4847, 0.4780],
         [0.4862, 0.4760],
         [0.4875, 0.4745],
         [0.4889, 0.4737],
         [0.4905, 0.4740],
         [0.4916, 0.4750],
         [0.4928, 0.4773],
         [0.4937, 0.4802],
         [0.4974, 0.4858],
         [0.5007, 0.4906],
         [0.5040, 0.4951],
         [0.5068, 0.4987],
         [0.5095, 0.5018],
         [0.5119, 0.5043],
         [0.5138, 0.5059],
         [0.5157, 0.5071],
         [0.5173, 0.5077],
         [0.5186, 0.5075],
         [0.5196, 0.5066],
         [0.5205, 0.5053],
         [0.5210, 0.5030],
         [0.5213, 0.5002],
         [0.5214, 0.4968],
         [0.5211, 0.4925],
         [0.5188, 0.4892],
         [0.5135, 0.4863],
         [0.5087, 0.4831],
         [0.5051, 0.4800],
         [0.5020, 0.4768],
         [0.4997, 0.4737],
         [0.4980, 0.4708],
         [0.4966, 0.4680],
         [0.4956, 0.4654],
         [0.4951, 0.4633],
         [0.4947, 0.4615],
         [0.4943, 0.4601],
         [0.4942, 0.4594],
         [0.4939, 0.4592],
         [0.4934, 0.4597],
         [0.4927, 0.4608],
         [0.4915, 0.4626],
         [0.4915, 0.4617],
         [0.4917, 0.4604],
         [0.4920, 0.4604],
         [0.4924, 0.4611],
         [0.4930, 0.4627],
         [0.4935, 0.4648],
         [0.4942, 0.4677],
         [0.4952, 0.4711],
         [0.4964, 0.4751],
         [0.4979, 0.4794],
         [0.4996, 0.4840],
         [0.5014, 0.4886],
         [0.5036, 0.4933],
         [0.5063, 0.4983],
         [0.5091, 0.5029],
         [0.5125, 0.5076]]], requires_grad=True)  # 模型的输出
target = torch.tensor([[[0.0067, 0.1877],
         [0.0121, 0.1877],
         [0.0176, 0.1878],
         [0.0232, 0.1879],
         [0.0288, 0.1880],
         [0.0345, 0.1881],
         [0.0402, 0.1882],
         [0.0459, 0.1883],
         [0.0517, 0.1886],
         [0.0575, 0.1888],
         [0.0633, 0.1891],
         [0.0691, 0.1894],
         [0.0749, 0.1897],
         [0.0808, 0.1901],
         [0.0866, 0.1906],
         [0.0924, 0.1910],
         [0.0982, 0.1915],
         [0.1040, 0.1920],
         [0.1098, 0.1927],
         [0.1155, 0.1934],
         [0.1211, 0.1940],
         [0.1267, 0.1948],
         [0.1323, 0.1956],
         [0.1379, 0.1965],
         [0.1433, 0.1975],
         [0.1487, 0.1984],
         [0.1540, 0.1995],
         [0.1593, 0.2007],
         [0.1645, 0.2019],
         [0.1696, 0.2031],
         [0.1745, 0.2045],
         [0.1794, 0.2059],
         [0.1842, 0.2074],
         [0.1888, 0.2089],
         [0.1933, 0.2105],
         [0.1978, 0.2123],
         [0.2021, 0.2141],
         [0.2062, 0.2159],
         [0.2101, 0.2178],
         [0.2140, 0.2199],
         [0.2177, 0.2221],
         [0.2213, 0.2243],
         [0.2246, 0.2265],
         [0.2279, 0.2290],
         [0.2308, 0.2314],
         [0.2337, 0.2341],
         [0.2364, 0.2368],
         [0.2387, 0.2395],
         [0.2410, 0.2424],
         [0.2431, 0.2455],
         [0.2440, 0.2469],
         [0.2449, 0.2620],
         [0.2458, 0.2773],
         [0.2466, 0.2923],
         [0.2473, 0.3075],
         [0.2479, 0.3226],
         [0.2486, 0.3377],
         [0.2491, 0.3529],
         [0.2497, 0.3680],
         [0.2502, 0.3832],
         [0.2506, 0.3984],
         [0.2509, 0.4134],
         [0.2513, 0.4286],
         [0.2516, 0.4437],
         [0.2519, 0.4589],
         [0.2521, 0.4740],
         [0.2523, 0.4892],
         [0.2524, 0.5042],
         [0.2527, 0.5195],
         [0.2528, 0.5347],
         [0.2528, 0.5496],
         [0.2529, 0.5648],
         [0.2530, 0.5799],
         [0.2530, 0.5951],
         [0.2531, 0.6102],
         [0.2531, 0.6253],
         [0.2531, 0.6405],
         [0.2532, 0.6557],
         [0.2532, 0.6709],
         [0.2532, 0.6861],
         [0.2532, 0.7012],
         [0.2532, 0.7163],
         [0.2533, 0.7316],
         [0.2533, 0.7466],
         [0.2533, 0.7616],
         [0.2534, 0.7770],
         [0.2535, 0.7922],
         [0.2535, 0.8073],
         [0.2536, 0.8223],
         [0.2537, 0.8374],
         [0.2539, 0.8528],
         [0.2541, 0.8680],
         [0.2542, 0.8829],
         [0.2545, 0.8984],
         [0.2547, 0.9131],
         [0.2550, 0.9285],
         [0.2554, 0.9438],
         [0.2557, 0.9587],
         [0.2561, 0.9737],
         [0.2566, 0.9892]],

        [[0.9932, 0.2135],
         [0.9836, 0.2135],
         [0.9741, 0.2135],
         [0.9639, 0.2133],
         [0.9536, 0.2131],
         [0.9427, 0.2127],
         [0.9317, 0.2123],
         [0.9204, 0.2119],
         [0.9092, 0.2114],
         [0.8977, 0.2109],
         [0.8859, 0.2104],
         [0.8737, 0.2098],
         [0.8618, 0.2093],
         [0.8496, 0.2088],
         [0.8375, 0.2083],
         [0.8252, 0.2078],
         [0.8128, 0.2074],
         [0.8003, 0.2070],
         [0.7884, 0.2068],
         [0.7761, 0.2066],
         [0.7637, 0.2065],
         [0.7518, 0.2065],
         [0.7399, 0.2067],
         [0.7280, 0.2070],
         [0.7164, 0.2074],
         [0.7049, 0.2079],
         [0.6937, 0.2087],
         [0.6828, 0.2097],
         [0.6721, 0.2108],
         [0.6617, 0.2121],
         [0.6516, 0.2136],
         [0.6417, 0.2153],
         [0.6324, 0.2174],
         [0.6232, 0.2196],
         [0.6145, 0.2220],
         [0.6064, 0.2249],
         [0.5987, 0.2279],
         [0.5914, 0.2313],
         [0.5845, 0.2349],
         [0.5782, 0.2388],
         [0.5727, 0.2432],
         [0.5675, 0.2478],
         [0.5629, 0.2527],
         [0.5591, 0.2582],
         [0.5556, 0.2638],
         [0.5532, 0.2700],
         [0.5513, 0.2765],
         [0.5500, 0.2833],
         [0.5494, 0.2906],
         [0.5500, 0.2985],
         [0.5503, 0.3025],
         [0.5501, 0.3165],
         [0.5503, 0.3306],
         [0.5502, 0.3445],
         [0.5503, 0.3586],
         [0.5501, 0.3725],
         [0.5501, 0.3865],
         [0.5501, 0.4006],
         [0.5502, 0.4146],
         [0.5503, 0.4287],
         [0.5503, 0.4427],
         [0.5502, 0.4567],
         [0.5503, 0.4707],
         [0.5504, 0.4847],
         [0.5505, 0.4988],
         [0.5506, 0.5128],
         [0.5507, 0.5268],
         [0.5507, 0.5407],
         [0.5510, 0.5549],
         [0.5511, 0.5689],
         [0.5511, 0.5827],
         [0.5513, 0.5968],
         [0.5515, 0.6108],
         [0.5516, 0.6248],
         [0.5518, 0.6389],
         [0.5520, 0.6528],
         [0.5522, 0.6669],
         [0.5525, 0.6810],
         [0.5528, 0.6951],
         [0.5531, 0.7091],
         [0.5533, 0.7231],
         [0.5535, 0.7371],
         [0.5540, 0.7513],
         [0.5541, 0.7651],
         [0.5543, 0.7790],
         [0.5548, 0.7933],
         [0.5552, 0.8074],
         [0.5555, 0.8214],
         [0.5558, 0.8352],
         [0.5562, 0.8492],
         [0.5567, 0.8635],
         [0.5571, 0.8775],
         [0.5574, 0.8913],
         [0.5580, 0.9057],
         [0.5582, 0.9193],
         [0.5588, 0.9336],
         [0.5594, 0.9477],
         [0.5597, 0.9615],
         [0.5602, 0.9754],
         [0.5609, 0.9898]],

        [[0.7793, 0.0033],
         [0.7816, 0.0036],
         [0.7834, 0.0038],
         [0.7855, 0.0040],
         [0.7879, 0.0043],
         [0.7899, 0.0045],
         [0.7921, 0.0047],
         [0.7944, 0.0049],
         [0.7965, 0.0051],
         [0.7985, 0.0053],
         [0.8005, 0.0054],
         [0.8026, 0.0056],
         [0.8048, 0.0058],
         [0.8072, 0.0060],
         [0.8090, 0.0061],
         [0.8111, 0.0063],
         [0.8134, 0.0064],
         [0.8156, 0.0066],
         [0.8178, 0.0067],
         [0.8199, 0.0069],
         [0.8221, 0.0070],
         [0.8240, 0.0071],
         [0.8262, 0.0072],
         [0.8283, 0.0073],
         [0.8306, 0.0075],
         [0.8328, 0.0076],
         [0.8349, 0.0077],
         [0.8371, 0.0078],
         [0.8393, 0.0079],
         [0.8414, 0.0080],
         [0.8436, 0.0081],
         [0.8455, 0.0081],
         [0.8479, 0.0082],
         [0.8499, 0.0083],
         [0.8520, 0.0084],
         [0.8545, 0.0085],
         [0.8566, 0.0085],
         [0.8586, 0.0086],
         [0.8609, 0.0087],
         [0.8630, 0.0087],
         [0.8649, 0.0088],
         [0.8674, 0.0088],
         [0.8693, 0.0089],
         [0.8716, 0.0089],
         [0.8737, 0.0090],
         [0.8760, 0.0090],
         [0.8780, 0.0091],
         [0.8802, 0.0091],
         [0.8824, 0.0092],
         [0.8844, 0.0092],
         [0.8866, 0.0092],
         [0.8889, 0.0093],
         [0.8910, 0.0093],
         [0.8932, 0.0093],
         [0.8955, 0.0094],
         [0.8976, 0.0094],
         [0.8999, 0.0094],
         [0.9019, 0.0095],
         [0.9043, 0.0095],
         [0.9062, 0.0095],
         [0.9086, 0.0096],
         [0.9108, 0.0096],
         [0.9128, 0.0096],
         [0.9153, 0.0096],
         [0.9174, 0.0097],
         [0.9192, 0.0097],
         [0.9215, 0.0097],
         [0.9238, 0.0098],
         [0.9257, 0.0098],
         [0.9282, 0.0098],
         [0.9304, 0.0098],
         [0.9326, 0.0099],
         [0.9347, 0.0099],
         [0.9368, 0.0099],
         [0.9391, 0.0100],
         [0.9412, 0.0100],
         [0.9432, 0.0100],
         [0.9455, 0.0101],
         [0.9475, 0.0101],
         [0.9501, 0.0102],
         [0.9522, 0.0102],
         [0.9544, 0.0102],
         [0.9565, 0.0103],
         [0.9586, 0.0103],
         [0.9606, 0.0104],
         [0.9627, 0.0104],
         [0.9653, 0.0105],
         [0.9672, 0.0105],
         [0.9692, 0.0106],
         [0.9715, 0.0107],
         [0.9738, 0.0107],
         [0.9761, 0.0108],
         [0.9783, 0.0109],
         [0.9803, 0.0109],
         [0.9823, 0.0110],
         [0.9847, 0.0111],
         [0.9865, 0.0112],
         [0.9887, 0.0112],
         [0.9912, 0.0113],
         [0.9932, 0.0114]]])  # 真实标签

# 计算 L1 损失
loss = F.l1_loss(input, target, )    #  sum 182.25799560546875
# print(f"L1 Loss: {loss.item()}")

er = range(0, 6, 2)
print(er[2])