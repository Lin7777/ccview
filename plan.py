import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.font_manager import FontProperties

# 地点和连接
places = [
    "长沙W酒店", "岳麓山", "岳麓书院", "橘子洲", "黄兴南路步行街",
    "火宫殿", "解放西路", "长沙岳麓万枫酒店", "太平街",
    "湖南省博物馆", "梅溪湖国际文化艺术中心", "梅溪湖公园",
    "天心阁", "爱晚亭", "长沙黄花国际机场"
]

connections = [
    ("长沙W酒店", "岳麓山"),
    ("岳麓山", "岳麓书院"),
    ("岳麓书院", "橘子洲"),
    ("橘子洲", "黄兴南路步行街"),
    ("黄兴南路步行街", "火宫殿"),
    ("火宫殿", "解放西路"),
    ("解放西路", "长沙岳麓万枫酒店"),
    ("长沙岳麓万枫酒店", "太平街"),
    ("太平街", "湖南省博物馆"),
    ("湖南省博物馆", "梅溪湖国际文化艺术中心"),
    ("梅溪湖国际文化艺术中心", "梅溪湖公园"),
    ("梅溪湖公园", "解放西路"),
    ("长沙岳麓万枫酒店", "天心阁"),
    ("天心阁", "爱晚亭"),
    ("爱晚亭", "长沙黄花国际机场")
]

# 创建图
G = nx.Graph()
G.add_nodes_from(places)
G.add_edges_from(connections)

# 设置位置
pos = {
    "长沙W酒店": (0, 10), "岳麓山": (2, 8), "岳麓书院": (3, 7),
    "橘子洲": (4, 8), "黄兴南路步行街": (6, 9),
    "火宫殿": (7, 10), "解放西路": (8, 11),
    "长沙岳麓万枫酒店": (9, 8), "太平街": (10, 7),
    "湖南省博物馆": (11, 6), "梅溪湖国际文化艺术中心": (12, 5),
    "梅溪湖公园": (13, 4), "天心阁": (14, 8), "爱晚亭": (15, 9),
    "长沙黄花国际机场": (16, 10)
}

# 指定中文字体
font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')

# 画图
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, labels={node: node for node in G.nodes()},
        node_size=3000, node_color="skyblue", font_size=10, font_family=font.get_name(), font_weight="bold", edge_color="gray")
plt.title("长沙旅游路线图", fontproperties=font)
plt.show()
