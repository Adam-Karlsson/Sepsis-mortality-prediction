import pandas as pd


# class meanValue:
#
#     def __init__(self, features):
#         self.meanValueDict = {}
#         self.pandasDF = pd.DataFrame
#         print("storlek", features.size)
#
#
#
#         # df_f = pd.DataFrame(features, columns=["importance"])
#         # self.meanValueDict = df_f.set_index('importance').T.to_dict('list')
#
#         # exempel på att lagra värden
#         # self.meanValueDict['EMS'] = 123
#         print("Hej")
#
#     def add(self, data, features):
#         print("adding")
#         df_f = pd.DataFrame(data, columns=["importance"])
#         dataDict = df_f.to_dict('list')
#         dataDict1 = df_f.to_dict('split')
#         dataDict2 = df_f.to_dict('records')
#         dataDict3 = df_f.to_dict('dict')
#         dataDict4 = df_f.to_dict('series')
#         dataDict5 = df_f.to_dict('index')
#
#         for key in self.meanValueDict:
#             print(key)
#         print("j")
#
#     def exportExcel(self):
#         print("export to excel")
#
#     def clear(self):
#         print("clear")