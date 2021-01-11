import yfinance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sn


# TODO pip install yfinance, scipy


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
        # TODO change days
        # simulation 1
        self.create_window(days=882, interval=10, number_simulation=100)
        profit = self.calculate_profit(start_money_arg=100, built_in=False)
        results = self.check_question(profit, built_in=False)
        print(results)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 B ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.create_window(days=882, interval=10, number_simulation=100)
        profit = self.calculate_profit(start_money_arg=100, built_in=True)
        results = self.check_question(profit, built_in=True)
        print(results)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 C1 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        # simulation 1
        self.create_window(days=882, interval=10, number_simulation=100)
        profit = self.calculate_profit(start_money_arg=100, built_in=False)
        results = self.check_question(profit, built_in=False)
        print(results)

        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 C2 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

        self.create_window(days=882, interval=10, number_simulation=100)
        profit = self.calculate_profit(start_money_arg=100, built_in=True)
        results = self.check_question(profit, built_in=True)
        print(results)

    def get_data(self):
        """
        data from Yahoo’s finance website.
        """
        for t in self.portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2021-1-1")
            data[f'price_{name}'] = data['Close']
            # TODO check multiply in 100
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

        cov_matrix = self.tickets[["return_VMC", "return_EMR", "return_CSX", "return_UNP"]].cov()
        print(f"Covariance Close: \n{cov_matrix}")
        sn.heatmap(cov_matrix, annot=True, fmt='g')
        plt.savefig("File/covMatrix/cov_matrix_returns.png")

    @staticmethod
    def show_price(ticker):
        ticker.plot()
        plt.xlabel("Date")
        plt.ylabel("Close")
        plt.title(f"{ticker.name} Price data")
        plt.savefig("File/Price/" + ticker.name + ".png")

    @staticmethod
    def calculate_returns(ticker):
        plt.figure(figsize=(10, 8))
        plt.hist(ticker, density=True)
        plt.title(f"Histogram - {ticker.name} daily returns data")
        plt.xlabel("Daily returns %")
        plt.ylabel("Percent")
        plt.savefig("File/DailyReturns/" + ticker.name + ".png")

    def remove_future(self):
        self.tickets = pd.DataFrame({})
        self.return_tickets = pd.DataFrame({})
        for t in self.portfolio_composition:
            name = t[0]
            ticker = yfinance.Ticker(name)
            data = ticker.history(interval="1d", start="1980-01-01", end="2018-03-26")
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
                # TODO check np.random
                rand_index = np.random.randint(low=0, high=len(self.return_tickets) - interval)
                random_days = self.return_tickets[rand_index: rand_index + interval]
                window_add.extend(random_days.values)
            self.windows[f"{rep}_VMC"] = np.array(window_add)[:, 0]
            self.windows[f"{rep}_EMR"] = np.array(window_add)[:, 1]
            self.windows[f"{rep}_CSX"] = np.array(window_add)[:, 2]
            self.windows[f"{rep}_UNP"] = np.array(window_add)[:, 3]

    def calculate_profit(self, start_money_arg=100, built_in=False):
        flag_over = False
        profit = pd.DataFrame(columns=['VMC', 'EMR', 'CSX', 'UNP'])
        # for all simulation
        for rep in range(self.windows.shape[1] // 4):
            window_check = self.windows[[f"{rep}_VMC", f"{rep}_EMR", f"{rep}_CSX", f"{rep}_UNP"]]
            list_add = []
            for ticket in window_check:
                name = ticket.split("_")[1]
                # init
                end_money = start_money_arg
                start_money = start_money_arg
                for timestamp in range(len(window_check[ticket])):
                    over = (end_money - start_money) / start_money
                    if (window_check[ticket][timestamp] > 0.36 or over < 0.36) and built_in:
                        flag_over = True
                        break
                    else:  # <= 0.36 / !built_in
                        end_money += end_money * window_check[ticket][timestamp]
                if flag_over:
                    list_add.append(0.02)
                else:
                    list_add.append((end_money - start_money) / start_money)

            profit = profit.append({"VMC": list_add[0], "EMR": list_add[1], "CSX": list_add[2], "UNP": list_add[3]},
                                   ignore_index=True)
        return profit

    def check_question(self, profit, built_in=False):
        results = {}
        for key in profit.keys():
            template = {"Name Stock": key,
                        "Loss (<0)": np.round(np.sum(profit[key] < 0), 2),
                        "0% profit": np.round(np.sum(profit[key] == 0), 2),
                        "2% profit": np.round(np.sum(profit[key] == 0.02), 2),
                        "(2%, 20%]": np.round(np.sum(profit[key][((0.2 > profit[key]) & (profit[key] < 0.2))]), 2),
                        "(20%, 36%)": np.round(np.sum(profit[key][((0.2 > profit[key]) & (profit[key] < 0.36))]), 2),
                        "Mean": np.round(profit[key].mean(), 2)}
            results[key] = template
        return results


if __name__ == '__main__':
    simulation = Simulation()
    simulation.main()
