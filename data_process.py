import pandas as pd
from sqlalchemy import create_engine
import time
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.cm as cm
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from dotenv import load_dotenv
import os
load_dotenv()

Conn_1 = os.getenv('Conn1')
Conn_2 = os.getenv('Conn2')
old_data = os.getenv('Old_data')
change_data = os.getenv('Change_data')
tran_data = os.getenv('Tran_data')

def create_table(engine,files):
    df = pd.read_csv(files,encoding='cp949',low_memory=False)
    print(df)
    # if_exists 空值給值
    start_time = time.time()
    try:
        df.to_sql('gamedata',index=False,con=engine,if_exists='replace', chunksize=50000)
    except Exception as e:
        print('e:',e)
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))

#修改欄位
def update_str(list_index,list_index_1,listPosition,gamelist):
    for i in list_index_1:
        if gamelist['teamPosition'][i] == None:
            pass
        else:
            remove_str = gamelist['teamPosition'][i]
            listPosition.remove(remove_str)
    if len(listPosition) == 0:
        pass
    else:
        #確認更新資訊
        print('listPosition :',listPosition[0])
        print('list_index: ',list_index)
        print('list_index2: ',gamelist['gameno'][list_index])
        print('list_index3: ',gamelist['participantId'][list_index])
        print('list_index4: ',gamelist['teamPosition'][list_index])
        gameno = gamelist['gameno'][list_index]
        participantId = gamelist['participantId'][list_index]
        update_str = f'UPDATE game.gamedata SET teamPosition = "{listPosition[0]}"  Where gameno = {gameno} and participantId = {participantId} ;'
        cnx.execute(update_str)

def columns_change(cnx):
    sum = 0
    gamelist = {}
    gamelist['gameno'] = []
    gamelist['participantId'] = []
    gamelist['teamPosition'] = []
    test_query = cnx.execute("select no,gameno,participantId,teamPosition from game.gamedata where no in (select no from game.gamedata where teamPosition IS NULL);")
    # 找尋團隊位置空值的欄位
    for i in test_query:
        #湊齊一個房間全部資訊
        listPosition = ['JUNGLE','MIDDLE','BOTTOM','TOP','UTILITY']
        sum = sum + 1
        if sum < 10:
            gamelist['gameno'].append(i[1])
            gamelist['participantId'].append(i[2])
            gamelist['teamPosition'].append(i[3])
        elif sum == 10:
            gamelist['gameno'].append(i[1])
            gamelist['participantId'].append(i[2])
            gamelist['teamPosition'].append(i[3])
            if gamelist['teamPosition'] == None:
                print(gamelist['teamPosition'])
            list_index = lists(gamelist['teamPosition'])
            print('list_index:',list_index)
            # 查詢空值位置並做資料更新
            if list_index < 5:
                list_index_1 = [0,1,2,3,4]
                update_str(list_index,list_index_1,listPosition,gamelist)
                gamelist['gameno'].clear()
                gamelist['participantId'].clear()
                gamelist['teamPosition'].clear()
            elif list_index >4 and list_index < 11:
                list_index_1 = [5,6,7,8,9]
                update_str(list_index,list_index_1,listPosition,gamelist)
                gamelist['gameno'].clear()
                gamelist['participantId'].clear()
                gamelist['teamPosition'].clear()
            sum = 0
    
def into_table_15(cnx,cnx2,index):
    test_query = cnx.execute(f"SELECT CreationTime,teamId,teamPosition,kills,deaths,assists,visionScore,baronKills,bountyLevel,champLevel,championName,damageDealtToBuildings,damageDealtToObjectives,dragonKills,firstBloodKill,firstTowerKill,goldEarned,inhibitorKills,inhibitorsLost,longestTimeSpentLiving,neutralMinionsKilled,timePlayed,totalDamageDealtToChampions,totalHealsOnTeammates,totalMinionsKilled,totalTimeCCDealt FROM game.gamedata  where {index}; ")
    r = []
    lists = []
    # query 資料拼出轉換所需的字串
    for i in test_query:
        date_ = datetime.strptime(i[0], '%Y-%m-%d %H:%M:%S')
        r.append(date_)
        r.append(int(i[1]))
        r.append(i[2])
        r.append(int(i[3]))
        r.append(int(i[4]))
        r.append(int(i[5]))
        r.append(int(i[6]))
        r.append(int(i[7]))
        r.append(int(i[8]))
        r.append(int(i[9]))
        r.append(i[10])
        r.append(int(i[11]))
        r.append(int(i[12]))
        r.append(int(i[13]))
        r.append(int(i[14]))
        r.append(int(i[15]))
        r.append(int(i[16]))
        r.append(int(i[17]))
        r.append(int(i[18])) 
        r.append(int(i[19]))
        r.append(int(i[20]))
        r.append(int(i[21]))
        r.append(int(i[22]))
        r.append(int(i[23]))
        r.append(int(i[24]))
        r.append(int(i[25]))
        lists.append(r)
        r = []
    #    [[0,15],[0,16]] [0,15]
    sum_2 = 0
    for i in lists:
        if sum_2 == 0: 
            str =f'("{i[0]}",{i[1]},"{i[2]}",{i[3]},{i[4]},{i[5]},{i[6]},{i[7]},{i[8]},{i[9]},"{i[10]}",{i[11]},{i[12]},{i[13]},{i[14]},{i[15]},{i[16]},{i[17]},{i[18]},{i[19]},{i[20]},{i[21]},{i[22]},{i[23]},{i[24]},{i[25]}),'
            sum_2 = sum_2 + 1
            
        else:
            str1 =f'("{i[0]}",{i[1]},"{i[2]}",{i[3]},{i[4]},{i[5]},{i[6]},{i[7]},{i[8]},{i[9]},"{i[10]}",{i[11]},{i[12]},{i[13]},{i[14]},{i[15]},{i[16]},{i[17]},{i[18]},{i[19]},{i[20]},{i[21]},{i[22]},{i[23]},{i[24]},{i[25]}),'
            str = str + str1
            sum_2 = sum_2+1
            print('insert into :',str1)
    
    
   
    insert_into = '''
                INSERT INTO GP_B_SRP_15_P_CT_ML_Trans_Reg (CreationTime ,teamId ,teamPosition ,kills ,deaths ,assists ,visionScore ,baronKills ,bountyLevel ,champLevel ,championName ,damageDealtToBuildings ,damageDealtToObjectives ,dragonKills ,firstBloodKill ,firstTowerKill ,goldEarned ,inhibitorKills ,inhibitorsLost ,longestTimeSpentLiving ,neutralMinionsKilled ,timePlayed ,totalDamageDealtToChampions ,totalHealsOnTeammates ,totalMinionsKilled ,totalTimeCCDealt ) VALUES '''
    insert_into = insert_into + str[:-1]
    test_query = cnx2.execute(insert_into)
    

# 轉換時間(分割一小時為1)
def time_change(new_time):
    if new_time < 3601:
        num = 0
    elif new_time > 3600 and new_time <= 7200:
        num = 1
    elif new_time > 7200 and new_time <= 10800:
        num = 2
    elif new_time > 10800 and new_time <= 14400:
        num = 3
    elif new_time > 14400 and new_time <= 18000:
        num = 4
    elif new_time > 18000 and new_time <= 21600:
        num = 5
    elif new_time > 21600 and new_time <= 25200:
        num = 6
    elif new_time > 25200 and new_time <= 28800:
        num = 7
    elif new_time > 28800 and new_time <= 32400:
        num = 8
    elif new_time > 32400 and new_time <= 36000:
        num = 9
    elif new_time > 36000 and new_time <= 39600:
        num = 10
    elif new_time > 39600 and new_time <= 43200:
        num = 11
    elif new_time > 43200 and new_time <= 46800:
        num = 12
    elif new_time > 46800 and new_time <= 50400:
        num = 13
    elif new_time > 50400 and new_time <= 54000:
        num = 14
    elif new_time > 54000 and new_time <= 57600:
        num = 15
    elif new_time > 57600 and new_time <= 61200:
        num = 16
    elif new_time > 61200 and new_time <= 64800:
        num = 17
    elif new_time > 64800 and new_time <= 68400:
        num = 18
    elif new_time > 68400 and new_time <= 72000:
        num = 19
    elif new_time > 72000 and new_time <= 75600:
        num = 20
    elif new_time > 75600 and new_time <= 79200:
        num = 21
    elif new_time > 79200 and new_time <= 82800:
        num = 22
    elif new_time > 82800:
        num = 23
    return num
def csv_to(data):
    
    #自行定位轉換指定位置
    #英雄位置
    championName_list = ['Alistar', 'Annie', 'Ashe', 'FiddleSticks', 'Jax',
    'Kayle', 'MasterYi', 'Morgana', 'Nunu', 'Ryze',
    'Sion', 'Sivir', 'Soraka', 'Teemo', 'Tristana',
    'TwistedFate', 'Warwick', 'Singed', 'Zilean', 'Evelynn',
    'Tryndamere', 'Twitch', 'Karthus', 'Amumu', 'Chogath',
    'Anivia', 'Rammus', 'Veigar', 'Kassadin', 'Gangplank',
    'Taric', 'Blitzcrank', 'DrMundo', 'Janna', 'Malphite',
    'Corki', 'Katarina', 'Nasus', 'Heimerdinger', 'Shaco',
    'Udyr', 'Nidalee', 'Poppy', 'Gragas', 'Pantheon',
    'Mordekaiser', 'Ezreal', 'Shen', 'Kennen', 'Garen',
    'Akali', 'Malzahar', 'Olaf', 'KogMaw', 'XinZhao',
    'Vladimir', 'Galio', 'Urgot', 'MissFortune', 'Sona',
    'Swain', 'Lux', 'Leblanc', 'Irelia', 'Trundle',
    'Cassiopeia', 'Caitlyn', 'Renekton', 'Karma', 'Maokai',
    'JarvanIV', 'Nocturne', 'LeeSin', 'Brand', 'Rumble',
    'Vayne', 'Orianna', 'Yorick', 'Leona', 'MonkeyKing',
    'Skarner', 'Talon', 'Riven', 'Xerath', 'Graves',
    'Shyvana', 'Fizz', 'Volibear', 'Ahri', 'Viktor',
    'Sejuani', 'Ziggs', 'Nautilus', 'Fiora', 'Lulu',
    'Hecarim', 'Varus', 'Darius', 'Draven', 'Jayce',
    'Zyra', 'Diana', 'Rengar', 'Syndra', 'Khazix',
    'Elise', 'Zed', 'Nami', 'Vi', 'Thresh', 'Quinn',
    'Zac', 'Lissandra', 'Aatrox', 'Lucian', 'Jinx',
    'Yasuo', 'Velkoz', 'Braum', 'Gnar', 'Azir',
    'Kalista', 'RekSai', 'Bard', 'Ekko', 'TahmKench',
    'Kindred', 'Illaoi', 'Jhin', 'AurelionSol', 'Taliyah',
    'Kled', 'Ivern', 'Camille', 'Rakan', 'Xayah', 'Kayn',
    'Ornn', 'Zoe', 'Kaisa', 'Pyke', 'Neeko', 'Sylas',
    'Yuumi', 'Qiyana', 'Senna', 'Aphelios', 'Sett',
    'Lillia', 'Yone', 'Samira', 'Seraphine', 'Rell',
    'Viego', 'Gwen', 'Akshan', 'Vex', 'Zeri', 'Renata',
    'Belveth', 'Nilah', "K'Sante"]

    #線路
    pos_list = ['TOP','JUNGLE','MIDDLE','BOTTOM','UTILITY']

    start_time = time.time()

    datas = {}
    
    datas['teamPositions'] = []
    datas['championNames'] = []
    datas['CreationTimes'] = []
    #時間處理
    for i in data['CreationTime']:
        # print(i)
        # 2022-07-02 00:00:11
        year = i[:-9]
        struct_time = time.strptime(i, "%Y-%m-%d %H:%M:%S")
        struct_time_2 = time.strptime(year, "%Y-%m-%d")
        time_stamp = int(time.mktime(struct_time))
        time_stamp_2 = int(time.mktime(struct_time_2))
        new_time = time_stamp-time_stamp_2
        datas['CreationTimes'].append(new_time)
    times = datas['CreationTimes']
    #線路處理
    for i in data['teamPosition']:
        # print('teamPosition',i)
        datas['teamPositions'].append(pos_list.index(i)+1)

    #英雄處理 
    for i in data['championName']:
        datas['championNames'].append(championName_list.index(i)+1)

    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', -1)
    times = pd.DataFrame(datas)
    data = pd.concat([data,times], axis=1,ignore_index=False)
    # 置換位置
    order = ['CreationTime','CreationTimes','teamId','win','teamPosition','teamPositions','championName','championNames','kills','deaths','assists','visionScore','baronKills','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','dragonKills','firstBloodKill','firstTowerKill','goldEarned','inhibitorKills','inhibitorsLost','longestTimeSpentLiving','neutralMinionsKilled','timePlayed','totalDamageDealtToChampions','totalHealsOnTeammates','totalMinionsKilled','totalTimeCCDealt']
    data = data[order]


    #打亂
    # data = data.reindex(np.random.permutation(data.index))


    onehotencoder = OneHotEncoder(categories='auto')
    #teamPosition
    Group_ohe = np.array(data['teamPositions']).reshape((len(data['teamPositions']), 1))
    data_str_ohe=onehotencoder.fit_transform(Group_ohe).toarray()
    team_str = pd.DataFrame(data_str_ohe)

    Group_two = np.array(data['championNames']).reshape((len(data['championNames']), 1))
    data_str_two=onehotencoder.fit_transform(Group_two).toarray()
    champ_str = pd.DataFrame(data_str_two)

    data = pd.concat([data,team_str,champ_str], axis=1,ignore_index=False)

    #刪除欄位
    del data['teamPosition']
    del data['championName']
    del data['CreationTime']
    # del data['CreationTimes']
    del data['teamPositions']
    del data['championNames']
    data.to_csv('One_testData_gpA_srp15_All.csv',index=False)

    end_time = time.time()
def tree(data):
    data.head()
    data.columns = ['創建房間時間','團隊ID','輸贏','團隊位置','殺人數',
            '死亡數','助攻數','視野分數','擊殺巴龍','賞金級別','英雄等級',
            '英雄名稱','建築物造成的損害','地圖目標總傷害','擊殺小龍','首殺',
            '拿下首塔','獲得金錢','破壞的兵營數','被破壞的兵營數','最長的生存時間',
            '擊殺中立野怪','播放時間','對英雄總傷害','隊友總治療量 ','擊殺小兵',
            '總控制時間']
    feature_cols = ['創建房間時間','團隊ID','團隊位置','殺人數','死亡數',
                    '助攻數','視野分數','擊殺巴龍','賞金級別','英雄等級',
                    '英雄名稱','建築物造成的損害','地圖目標總傷害','擊殺小龍','首殺',
                    '拿下首塔','獲得金錢','破壞的兵營數','被破壞的兵營數','最長的生存時間',
                    '擊殺中立野怪','播放時間','對英雄總傷害','隊友總治療量 ','擊殺小兵',
                    '總控制時間']
    X = data[feature_cols] 
    y = data.輸贏
    # # del data['遊戲結束於早期投降']
    # # del data['遊戲結束於投降']
    # # del data['團隊早期投降']
    # # del data['破壞兵營的助攻']
    # # del data['破壞的兵營數']
    # del data['助攻地圖目標被偷']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test

    # Create Decision Tree classifer object
    clf = DecisionTreeClassifier(criterion="entropy",max_depth=4, random_state=42)  # entropy 為 C4.5 gini，即CART 預設gini

    # Train Decision Tree Classifer
    clf = clf.fit(X_train,y_train)

    #Predict the response for test dataset
    y_pred = clf.predict(X_test)


    print('X_test:',X_test)
    # Model Accuracy, how often is the classifier correct?
    print("R2:",metrics.accuracy_score(y_test, y_pred))

    from sklearn.tree import export_graphviz
    from six import StringIO 
    from IPython.display import Image  
    import pydotplus

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data,  
                    filled=True, rounded=True,
                    special_characters=True,feature_names = feature_cols,class_names=['0','1'])

    dot_data_val = dot_data.getvalue()
    dot_data_val = dot_data_val.replace('helvetica', 'MicrosoftYaHei')
    graph = pydotplus.graph_from_dot_data(dot_data_val)

    graph.write_png('DecisionTree6.png')
    Image(graph.create_png())
def kmeans_(files):

    data= pd.read_csv(files, header=0,index_col=0)
    # 1.baronKills,damageDealtToObjectives,dragonKills,neutralMinionsKilled 地圖物件控制率
    y = data['win'].values
    # print(data['baronKills'])
    # pdcolumns = ['baronKills','damageDealtToObjectives','dragonKills','neutralMinionsKilled']
    # data = data[pdcolumns]
    # print(data.columns)
    # 2.damageDealtToBuildings,firstTowerKill,inhibitorKills,inhibitorsLost 破壞建物達成率
    # 破壞建物達成率
    # pdcolumns = ['damageDealtToBuildings','firstTowerKill','inhibitorKills','inhibitorsLost']
    # data = data[pdcolumns]
    # print(data.columns)
    # 3.teamId,teamPositions,championNames,totalDamageDealtToChampions,totalHealsOnTeammates,totalTimeCCDealt 會戰定位
    # 會戰定位
    pdcolumns = ['teamPositions','championNames','totalDamageDealtToChampions','totalHealsOnTeammates','totalTimeCCDealt']
    data = data[pdcolumns]
    print(data.columns)
    # 4.kills,deaths,assists 個人戰績
    # pdcolumns = ['kills','deaths','assists']
    # data = data[pdcolumns]
    # print(data.columns)
    # print(data.columns)
    # 5.kills,assists,neutralMinionsKilled,totalMinionsKilled 經濟獲取效率
    # pdcolumns =['kills','assists','neutralMinionsKilled','totalMinionsKilled']
    # data = data[pdcolumns]
    # print(data.columns)
    # 6.baronKills,dragonKills 擊殺小龍
    # pdcolumns =['baronKills','dragonKills']
    # data = data[pdcolumns]
    
    print(data.columns)
    lists = data.values
    # print(type(lists))
    #輸入
    X = lists
    
    range_n_clusters = [2, 3, 4, 5, 6]

    for n_clusters in range_n_clusters:
        # 建立具有1行2列的子圖
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # 第一個子圖是輪廓圖
        # 輪廓係數的範圍為-0.1、1   
        ax1.set_xlim([-0.1, 1])

        # （n_clusters + 1）* 10用於在各個群集的輪廓圖之間插入空格，以明確劃分它們。
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # 使用n_clusters值和seed 10隨機初始集群。
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # Silhouette_score給出所有樣本的平均值。
        # 這可以透視形成的簇的密度和分離
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
            "The average silhouette_score is :", silhouette_avg)

        # 計算每個樣本的輪廓分數
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # 彙總屬於聚類i的樣本的輪廓分數，並對它們進行排序
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                            0, ith_cluster_silhouette_values,
                            facecolor=color, edgecolor=color, alpha=0.7)

            # 在輪廓圖的中間標註聚類編號
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # 下一個圖計算新的y_lower
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # 所有值的平均輪廓分數的垂直線
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # 清除y軸標籤/刻度
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 第二個圖顯示了實際形成的集群
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # 標記集群
        centers = clusterer.cluster_centers_
        # 在群集中心繪製白色圓圈
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("map_control Silhouette analysis for KMeans clustering on sample data "
                    "with n_clusters = %d" % n_clusters),
                    fontsize=14, fontweight='bold')

        plt.show()
        
def rfe_function(files):

    start = time.time()
    # 'CreationTimes','teamId','win','teamPositions','kills','deaths','assists','visionScore','baronKills','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','dragonKills','firstBloodKill','firstTowerKill','goldEarned','inhibitorKills','inhibitorsLost','longestTimeSpentLiving','neutralMinionsKilled','timePlayed','totalDamageDealtToChampions','totalHealsOnTeammates','totalMinionsKilled','totalTimeCCDealt'
    diabetes=pd.read_csv(files,header=0)
    #選取需要的欄位
    pdcolumns = ['win','CreationTimes','teamId','teamPositions','kills','deaths','assists','visionScore','baronKills','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','dragonKills','firstBloodKill','firstTowerKill','goldEarned','inhibitorKills','inhibitorsLost','longestTimeSpentLiving','neutralMinionsKilled','timePlayed','totalDamageDealtToChampions','totalHealsOnTeammates','totalMinionsKilled','totalTimeCCDealt']
    # pdcolumns = ['win','CreationTimes','deaths','assists','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','goldEarned','inhibitorsLost','longestTimeSpentLiving','timePlayed','totalDamageDealtToChampions','totalMinionsKilled']
    # pdcolumns = ['win','damageDealtToObjectives','goldEarned','inhibitorsLost','damageDealtToBuildings','bountyLevel','assists']
    # pdcolumns = ['win','inhibitorsLost','goldEarned','damageDealtToObjectives']
    diabetes = diabetes[pdcolumns]
    diabetes.info()
    # ['CreationTimes','deaths','assists','bountyLevel','damageDealtToBuildings','damageDealtToObjectives','goldEarned', 'inhibitorsLost','longestTimeSpentLiving','timePlayed','totalDamageDealtToChampions','totalMinionsKilled','totalTimeCCDealt']
    cols_to_norm = ['CreationTimes','teamId','teamPositions','kills','deaths','assists','visionScore','baronKills','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','dragonKills','firstBloodKill','firstTowerKill','goldEarned','inhibitorKills','inhibitorsLost','longestTimeSpentLiving','neutralMinionsKilled','timePlayed','totalDamageDealtToChampions','totalHealsOnTeammates','totalMinionsKilled','totalTimeCCDealt']
    # cols_to_norm = ['CreationTimes','deaths','assists','bountyLevel','champLevel','damageDealtToBuildings','damageDealtToObjectives','goldEarned','inhibitorsLost','longestTimeSpentLiving','timePlayed','totalDamageDealtToChampions','totalMinionsKilled']
    # cols_to_norm = ['damageDealtToObjectives','goldEarned','inhibitorsLost','damageDealtToBuildings','bountyLevel','assists']
    # cols_to_norm = ['damageDealtToObjectives','goldEarned','inhibitorsLost']
    diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min())*0.6+0.2)

    # labels y
    labels = diabetes['win']
    # x_data X
    x_data = diabetes[cols_to_norm]
    
    X_train, X_test, y_train, y_test = train_test_split(x_data,labels,test_size=0.33, random_state=81)

    y_test

    """# **特徵選擇的方法- -Wrapper**

    ## **邏輯迴歸+Sckitlearn逐步方法**
    """

    
    # 使用邏輯回歸作為模型
    # lin_reg = LogisticRegression(random_state=0)
    
    # clf = DecisionTreeClassifier(criterion="entropy",max_depth=6, random_state=42)
    lin_reg = DecisionTreeClassifier()
    # 選擇5個變量：可以更改並在模型中檢查其準確性
    rfe_mod = RFE(lin_reg,step=1)  # RFECV(lin_reg, step=1, cv=5) 
    X = X_train
    y = y_train
    names=pd.DataFrame(X_train.columns)
    myvalues=rfe_mod.fit(X,y) #to fit
    myvalues.support_  #The mask of selected features.
    myvalues.ranking_  #特徵排名，使得ranking_ [i]對應於第i個特徵的排名位置。選定的（即最佳估計）特徵被分配為等級1.
    rankings=pd.DataFrame(myvalues.ranking_) #Make it into data frame

    scored=pd.concat([names,rankings], axis=1)
    scored.columns = ["Feature", "Score"]
    mod_name = scored.sort_values(by=['Score']).head(13)
    print('mod_name',mod_name)
    #Concat and name columns
    ranked=pd.concat([names,rankings], axis=1)
    ranked.columns = ["Feature", "Rank"]
    ranked

    #選擇最重要(Only 1's)
    most_important = ranked.loc[ranked['Rank'] ==1] 
    print('most_important',most_important)
    most_important['Rank'].count()
    print('[Rank].count()',most_important['Rank'].count())
    lin_reg.fit(X_train,y_train) 
    #lin_reg.coef_

    """### **測試結果**"""

    # Filter:指定變數
    # lrX = X_test[['Number_pregnant','Glucose_concentration','BMI','Pedigree']]
    # lry = y_test
    # y_pred=lin_reg.predict(lrX)
    y_pred=lin_reg.predict(X_test)

    # y_pred =rfe_mod.predict(lrX)
    # accuracy = accuracy_score(y_test, y_pred)
    # print("Logic Regression Accuracy: %0.1f%% " % (accuracy * 100))

    print(classification_report(y_test,y_pred))



    """## **RandomForest**"""
    
    xgbc = XGBClassifier(        
            n_estimators=100,
            max_depth=6,
            learning_rate=0.18,
            random_state=81)

    xgbc.fit(X_train, y_train)
    RFtrain_score=xgbc.score(X_train, y_train)
    print('xgb train',RFtrain_score)

    """### **測試結果**"""

    RF_pred=xgbc.predict(X_test)
    # print(RF_pred)

    RFtest_score=xgbc.score(X_test,y_test)
    print('The R2 of XGB Classifier on testing set:', RFtest_score)
   
    print(classification_report(y_test,RF_pred))
    print(confusion_matrix(y_test,RF_pred))

# 連線1
engine = create_engine(Conn_1)
cnx = engine.connect()

# 連線2
engine2 = create_engine(Conn_2)
cnx2 = engine2.connect()

# Mysql匯出CSV
# SELECT * 
# INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/GP_B_SRP_15_P_CT_ML_Trans_Reg.csv'
# FIELDS TERMINATED BY ','
# ESCAPED BY "'"
# optionally enclosed  by '' 
# LINES terminated by '\n' from (select 'CreationTime','teamId','teamPosition','win','kills','deaths','assists','visionScore','baronKills','bountyLevel','champLevel','championName','damageDealtToBuildings','damageDealtToObjectives','dragonKills','firstBloodKill','firstTowerKill','goldEarned','inhibitorKills','inhibitorsLost','longestTimeSpentLiving','neutralMinionsKilled','timePlayed','totalDamageDealtToChampions','totalHealsOnTeammates','totalMinionsKilled','totalTimeCCDealt' union select CreationTime,teamId,teamPosition,win,kills,deaths,assists,visionScore,baronKills,bountyLevel,champLevel,championName,damageDealtToBuildings,damageDealtToObjectives,dragonKills,firstBloodKill,firstTowerKill,goldEarned,inhibitorKills,inhibitorsLost,longestTimeSpentLiving,neutralMinionsKilled,timePlayed,totalDamageDealtToChampions,totalHealsOnTeammates,totalMinionsKilled,totalTimeCCDealt  from GP_B_SRP_18_P_ML_Trans_Reg) b;

#匯入資料
create_table(engine,old_data)
#欄位修改
columns_change(cnx)
lists = ['no < 40000''no > 39999 and no < 80000','no > 79999 and no < 130000','no > 129999 and no < 180000','no > 179999 and no < 220000','no > 219999 and no < 260000','no > 259999 and no < 300000']
for index in lists:
    into_table_15(cnx,cnx2,index)
    
#轉換le或one
csv_to(change_data)

#決策樹
tree(tran_data)

#Kmeans轉換
kmeans_(tran_data)

# RFE
rfe_function(tran_data)