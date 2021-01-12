import statistics

import scipy
import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sn

np.random.seed(20)


class Simulation:

    def __init__(self):
        self.all_ticket = None
        self.tickets = pd.DataFrame({})
        self.return_tickets = pd.DataFrame({})
        self.windows = pd.DataFrame({})
        self.portfolio_composition = [("VMC", 0.25), ('EMR', 0.25), ('CSX', 0.25), ('UNP', 0.25)]

    def main(self):
        # 1 A
        self.get_data()
        # 1 B
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1 B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.statical()

        self.remove_future()

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 A ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # simulation 1
        self.create_window(days=880, interval=10, number_simulation=100)
        self.windows.to_csv('1_windows.csv', index=False)

        profit = self.calculate_profit( built_in=False)
        self.calculate("2A", profit)

        self.check_question(profit, '1', built_in=False)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.create_window(days=880, interval=10, number_simulation=100)
        self.windows.to_csv('2_windows.csv', index=False)

        profit = self.calculate_profit( built_in=True)
        self.calculate("2B", profit)

        self.check_question(profit, '2', built_in=True)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 C1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # simulation 1
        self.create_window(days=880, interval=10, number_simulation=100)
        self.windows.to_csv('3_windows.csv', index=False)

        profit = self.calculate_profit(built_in=False)
        self.calculate("2C1", profit)

        self.check_question(profit, '3', built_in=False)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 C2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.create_window(days=880, interval=10, number_simulation=100)
        self.windows.to_csv('4_windows.csv', index=False)
        profit = self.calculate_profit(built_in=True)
        self.calculate("2C2", profit)

        self.check_question(profit, '4', built_in=True)

    def get_data(self):
        """
        data from Yahoo’s finance website.
        """
        for t in self.portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2021-1-1")
            data[f'price_{name}'] = data['Close']
            data['return_%s' % (name)] = data['Close'].pct_change(1)
            self.tickets = self.tickets.join(data[['price_%s' % (name)]], how="outer").dropna()
            self.tickets = self.tickets.join(data[[f'return_{name}']], how="outer").dropna()
        self.return_tickets = self.tickets[["return_VMC", "return_EMR", "return_CSX", "return_UNP"]]

    def statical(self):
        for t in self.portfolio_composition:
            name = t[0]
            self.show_price(self.tickets[f"price_{name}"])
            self.calculate_returns(self.tickets['return_' + name])
            print(f"---------{'return_' + name}--------------")
            print(f"Mean: {self.tickets['return_' + name].mean()}")
            print(f"Std: {self.tickets['return_' + name].std()}")
            print(f"Correlation Close: {self.tickets['return_' + name].autocorr()}")
            print(f"Correlation Price: {self.tickets['price_' + name].autocorr()}")
        plt.clf()
        mask = np.zeros_like(self.tickets[["return_VMC", "return_EMR", "return_CSX", "return_UNP"]].cov(), dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        cov_matrix = self.tickets[["return_VMC", "return_EMR", "return_CSX", "return_UNP"]].cov()
        print(f"Covariance Close: \n{cov_matrix}")
        cmap = sn.diverging_palette(220, 20, as_cmap=True)
        sn.heatmap(cov_matrix, mask=mask, cmap=cmap, center=0,
            square=True, linewidths=.5, annot=True)
        plt.title("Covariance Matrix")
        plt.savefig("File_covMatrix_cov_matrix_returns.png")

    @staticmethod
    def show_price(ticker):
        ticker.plot()
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.title(f"{ticker.name} Price data")
        plt.savefig("File_Price_" + ticker.name + ".png")

    @staticmethod
    def calculate_returns(ticker):
        plt.figure(figsize=(10, 8))
        plt.hist(ticker, density=True)
        plt.title(f"Histogram - {ticker.name} daily returns data")
        plt.xlabel("Daily returns %")
        plt.ylabel("Percent")
        plt.savefig("File_Profit_" + ticker.name + ".png")

    def calculate(self, name, profit):
        plt.figure(figsize=(10, 8))
        plt.hist(profit, density=True)
        plt.title(f"Histogram - Profit {name}")
        plt.savefig("File_Profit_" + name + ".png")

    def remove_future(self):
        self.tickets = pd.DataFrame({})
        self.return_tickets = pd.DataFrame({})
        for t in self.portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2018-03-25")
            # TODO check multiply in 100
            data['return_%s' % (name)] = data['Close'].pct_change(1)
            self.tickets = self.tickets.join(data[[f'return_{name}']], how="outer").dropna()
        self.return_tickets = self.tickets[["return_VMC", "return_EMR", "return_CSX", "return_UNP"]]

    def create_window(self, days=880, interval=10, number_simulation=10):
        size_window = days // interval
        for rep in range(number_simulation):
            window_add = []
            for window in range(size_window):
                # found random index
                rand_index = np.random.randint(low=0, high=len(self.return_tickets) - interval)
                random_days = self.return_tickets[rand_index: rand_index + interval]
                window_add.extend(random_days.values)
            self.windows[f"{rep}_VMC"] = np.array(window_add)[:, 0]
            self.windows[f"{rep}_EMR"] = np.array(window_add)[:, 1]
            self.windows[f"{rep}_CSX"] = np.array(window_add)[:, 2]
            self.windows[f"{rep}_UNP"] = np.array(window_add)[:, 3]

    @staticmethod
    def cumprod(tickets):
        com = ((tickets + 1).cumprod()).to_frame()
        com = com.applymap(lambda x: (x - 1) * 100)
        return np.sum(com[com > 36]) > 0, com

    def calculate_profit(self, built_in=False):
        flag_over = False
        # profit = pd.DataFrame(columns=['VMC', 'EMR', 'CSX', 'UNP'])
        profit = []
        # for all simulation
        for rep in range(self.windows.shape[1] // 4):
            window_check = self.windows[[f"{rep}_VMC", f"{rep}_EMR", f"{rep}_CSX", f"{rep}_UNP"]]
            list_add = []
            for ticket in window_check:
                ans, com = self.cumprod(window_check[ticket])
                # name = ticket.split("_")[1]
                # # init
                # end_money = start_money_arg
                # start_money = start_money_arg
                # for timestamp in range(len(window_check[ticket])):
                #     over = (end_money - start_money) / start_money
                #     if (window_check[ticket][timestamp] > 0.36 or over > 0.36) and built_in:
                #         flag_over = True
                #         break
                #     else:  # <= 0.36 / !built_in
                #         end_money += end_money * window_check[ticket][timestamp]
                if ans.values[0] and built_in:
                    list_add.append(2)
                else:
                    # if built_in:
                    #     print(com[len(window_check[ticket]) - 1:len(window_check[ticket])].values[0])
                    # list_add.append(((end_money - start_money) / start_money) * 100)
                    list_add.append(com[len(window_check[ticket]) - 1:len(window_check[ticket])].values[0])

            # profit = profit.append({"VMC": list_add[0], "EMR": list_add[1], "CSX": list_add[2], "UNP": list_add[3]},
            #                        ignore_index=True)
            #                        ignore_index=True)
            if (list_add[0] + list_add[1] + list_add[2] + list_add[3]) / 4 < 0 and built_in:
                profit.append(0)
            else:
                add = (list_add[0] + list_add[1] + list_add[2] + list_add[3]) / 4
                if isinstance(add, float):
                    profit.append(add)
                else:
                    profit.append(add[0])
        return profit

    def check_question(self, profit, index, built_in=False):
        # results = {}
        df = pd.DataFrame(profit)
        df.to_csv(index + '.csv', header=False, index=False)

        std = df.std().values[0]
        mean = df.mean().values[0]
        median = df.median().values[0]
        print(f"Mean = {mean}")
        print(f"Median = {median}")
        print(f"Std = {std}")
        if built_in:
            print(f"0% profit {sum(i==0 for i in profit) / len(profit)}")
            print(f"2% profit {sum(i == 2 for i in profit) / len(profit)}")
        else:
            print(f"0% profit {sum(-0.5 < i <0.5 for i in profit) / len(profit)}")
            print(f"2% profit {sum(1.5 < i <2.5 for i in profit) / len(profit)}")

        print(f"(2%, 20%] {sum(2 < i <= 20 for i in profit)/len(profit)}")
        print(f"(20%, 36%){sum(20 < i < 36 for i in profit)/len(profit)}")

        sort = sorted(profit)
        sort = sort[6: len(sort) - 5]
        print(f"Confidence Interval {sort[0]} - {sort[len(sort) -1]}")

        # print(f"0% profit {scipy.stats.norm(mean, std).pdf(0)*100}")
        # print(f"2% profit {scipy.stats.norm(mean, std).pdf(2)*100}")
        # print(f"(2%, 20%] {(scipy.stats.norm(mean, std).cdf(20)-scipy.stats.norm(mean, std).cdf(2))*100}")
        # print(f"(20%, 36%){(scipy.stats.norm(mean, std).cdf(36)-scipy.stats.norm(mean, std).cdf(20))*100}")
        # print(f"Confidence Interval { scipy.stats.norm.interval(0.9, loc=mean, scale=std)}")
        # for key in profit.keys():
        #     template = {"Name Stock": key,
        #                 "Loss (<0)": np.round(np.sum(profit[key] < 0), 2),
        #                 "0% profit": np.round(np.sum(profit[key] == 0), 2),
        #                 "2% profit": np.round(np.sum(profit[key] == 0.02), 2),
        #                 "(2%, 20%]": np.round(np.sum(profit[key][((0.02 > profit[key]) & (profit[key] < 0.2))]), 2),
        #                 "(20%, 36%)": np.round(np.sum(profit[key][((0.2 > profit[key]) & (profit[key] < 0.36))]), 2),
        #                 "Mean": np.round(profit[key].mean(), 2)}
        #     results[key] = templateS


if __name__ == '__main__':
    simulation = Simulation()
    simulation.main()
