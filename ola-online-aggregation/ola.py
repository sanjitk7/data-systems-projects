from HLL import HyperLogLog
from typing import List, Any

import pandas as pd
import plotly.graph_objects as go


class OLA:
    def __init__(self, widget: go.FigureWidget):
        """
            Base OLA class.

            *****************************************
            * You do not have to modify this class. *
            *****************************************

            @param widget: The dynamically updating plotly plot.
        """
        self.widget = widget

    def process_slice(df_slice: pd.DataFrame) -> None:
        """
            Process a dataframe slice. To be implemented in inherited classes.
        """
        pass

    def update_widget(self, groups_list: List[Any], values_list: List[Any]) -> None:
        """
            Update the plotly widget with newest groupings and values.

            @param groups_list: List of groups.
            @param values_list: List of grouped values (e.g., grouped means/sums).
        """
        self.widget.data[0]['x'] = groups_list
        self.widget.data[0]['y'] = values_list


class AvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated mean of *mean_col*.
            This class is implemented for you as an example.

            @param mean_col: column to compute filtered mean for.
        """
        super().__init__(widget)
        self.mean_col = mean_col

        # Bookkeeping variables
        self.sum = 0
        self.count = 0

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running mean with a data frame slice.
        """
        self.sum += df_slice.sum()[self.mean_col]
        self.count += df_slice.count()[self.mean_col]

        # Update the plot. The mean should be put into a singleton list due to Plotly semantics.
        # Note: there is no x axis label since there is only one bar.
        self.update_widget([""], [self.sum / self.count])


class FilterAvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, filter_col: str, filter_value: Any, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated filtered mean of *mean_col*
            where *filter_col* is equal to *filter_value*.

            @param filter_col: column to filter on.
            @param filter_value: value to filter for, i.e., df[df[filter_col] == filter_value].
            @param mean_col: column to compute filtered mean for.
        """
        super().__init__(widget)
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.mean_col = mean_col
        self.sum = 0
        self.count = 0
        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running filtered mean with a dataframe slice.
        """
        # Implement me!
        self.sum += df_slice[df_slice[self.filter_col]==self.filter_value].sum()[self.mean_col]
        self.count += df_slice[df_slice[self.filter_col]==self.filter_value].count()[self.mean_col]

        # Update the plot. The filtered mean should be put into a singleton list due to Plotly semantics.
        # hint: self.update_widget([""], *estimated filtered mean of mean_col*)
        self.update_widget([""], [self.sum / self.count])


class GroupByAvgOla(OLA):
    def __init__(self, widget: go.FigureWidget, groupby_col: str, mean_col: str):
        """
            Class for performing OLA by incrementally computing the estimated grouped means of *mean_col*
            with *groupby_col* as groups.

            @param groupby_col: grouping column, i.e., df.groupby(groupby_col).
            @param mean_col: column to compute grouped means for.
        """
        super().__init__(widget)
        self.groupby_col = groupby_col
        self.mean_col = mean_col
        self.grouped_sum = pd.Series(dtype=float)
        self.grouped_count = pd.Series(dtype=float)

        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running grouped means with a dataframe slice.
        """
        current_slice_sum = df_slice.groupby(self.groupby_col).sum()[self.mean_col]
        current_slice_count = df_slice.groupby(self.groupby_col).count()[self.mean_col]
        
        # incrementally updating these (fill 0 to replace new group sum/count not having a val that exists prev - replace NaN)
        self.grouped_sum = self.grouped_sum.add(current_slice_sum, fill_value=0)
        self.grouped_count = self.grouped_count.add(current_slice_count, fill_value=0)
        
        updated_means = self.grouped_sum / self.grouped_count
        unique_groups = updated_means.index.tolist() #index - indexes in the series (created by groupby operation) are the catagories they are being grouped on


        # Update the plot
        # hint: self.update_widget(*list of groups*, *list of estimated group means of mean_col*)
        self.update_widget(unique_groups,updated_means)


class GroupBySumOla(OLA):
    def __init__(self, widget: go.FigureWidget, original_df_num_rows: int, groupby_col: str, sum_col: str):
        """
            Class for performing OLA by incrementally computing the estimated grouped sums of *sum_col*
            with *groupby_col* as groups.

            @param original_df_num_rows: number of rows in the original dataframe before sampling and slicing.
            @param groupby_col: grouping column, i.e., df.groupby(groupby_col).
            @param sum_col: column to compute grouped sums for.
        """
        super().__init__(widget)
        self.original_df_num_rows = original_df_num_rows
        self.groupby_col = groupby_col
        self.sum_col = sum_col
        self.grouped_sum = pd.Series(dtype=float)
        self.cumulative_df_slice_num_rows = 0 

        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running grouped sums with a dataframe slice.
        """
        current_slice_sum = df_slice.groupby(self.groupby_col).sum()[self.sum_col]
        
        # track how many rows have been processed so far to scale up not just the current one
        self.cumulative_df_slice_num_rows += len(df_slice)
        
        # sampling fraction - fraction of orignal data that the current slice represents
        sampling_fraction = self.cumulative_df_slice_num_rows / self.original_df_num_rows
        
        # incrementally updating these (fill 0 to replace new group sum/count not having a val that exists prev - replace NaN)
        self.grouped_sum = self.grouped_sum.add(current_slice_sum, fill_value=0)
        
        unique_groups = self.grouped_sum.index.tolist()
        
        # scale the cumulative sum calculated so far (not the current slice sum) -- this wont affect future sum aggregation because "self.grouped_sum" isnt scaled
        grouped_sum_estimate = self.grouped_sum / sampling_fraction
        
        # Update the plot
        # hint: self.update_widget(*list of groups*, *list of estimated grouped sums of sum_col*)
        self.update_widget(unique_groups,grouped_sum_estimate.tolist())


class GroupByCountOla(OLA):
    def __init__(self, widget: go.FigureWidget, original_df_num_rows: int, groupby_col: str, count_col: str):
        """
            Class for performing OLA by incrementally computing the estimated grouped non-null counts in *count_col*
            with *groupby_col* as groups.

            @param original_df_num_rows: number of rows in the original dataframe before sampling and slicing.
            @param groupby_col: grouping column, i.e., df.groupby(groupby_col).
            @param count_col: counting column.
        """
        super().__init__(widget)
        self.original_df_num_rows = original_df_num_rows
        self.groupby_col = groupby_col
        self.count_col = count_col
        self.grouped_count = pd.Series(dtype=float)
        self.cumulative_df_slice_num_rows = 0

        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running grouped counts with a dataframe slice.
        """
        current_slice_count = df_slice.groupby(self.groupby_col).count()[self.count_col]
        
        # track how many rows have been processed so far to scale up not just the current one
        self.cumulative_df_slice_num_rows += len(df_slice)
        
        # sampling fraction - fraction of orignal data that the current slice represents
        sampling_fraction = self.cumulative_df_slice_num_rows / self.original_df_num_rows
        
        # incrementally updating these (fill 0 to replace new group sum/count not having a val that exists prev - replace NaN)
        self.grouped_count = self.grouped_count.add(current_slice_count, fill_value=0)
        
        unique_groups = self.grouped_count.index.tolist()
        
        # scale the cumulative sum calculated so far (not the current slice sum) -- this wont affect future sum aggregation because "self.grouped_sum" isnt scaled
        grouped_count_estimate = self.grouped_count / sampling_fraction
        
        # Update the plot
        # hint: self.update_widget(*list of groups*, *list of estimated grouped counts of count_col*)
        self.update_widget(unique_groups,grouped_count_estimate.tolist())


class FilterDistinctOla(OLA):
    def __init__(self, widget: go.FigureWidget, filter_col: str, filter_value: Any, distinct_col: str):
        """
            Class for performing OLA by incrementally computing the estimated cardinality (distinct elements) *distinct_col*
            where *filter_col* is equal to *filter_value*.

            @param filter_col: column to filter on.
            @param filter_value: value to filter for, i.e., df[df[filter_col] == filter_value].
            @param distinct_col: column to compute cardinality for.
        """
        super().__init__(widget)
        self.filter_col = filter_col
        self.filter_value = filter_value
        self.distinct_col = distinct_col

        # HLL for estimating cardinality. Don't modify the parameters; the autograder relies on it.
        # IMPORTANT: Please convert your data to the String type before adding to the HLL, i.e., self.hll.add(str(data))
        self.hll = HyperLogLog(p=2, seed=123456789)

        # Put any other bookkeeping class variables you need here...

    def process_slice(self, df_slice: pd.DataFrame) -> None:
        """
            Update the running filtered cardinality with a dataframe slice.
        """
        filtered_distinct_col = df_slice[df_slice[self.filter_col]==self.filter_value][self.distinct_col]
        
        for val in filtered_distinct_col:
            self.hll.add(str(val))

        # Update the plot. The filtered cardinality should be put into a singleton list due to Plotly semantics.
        # hint: self.update_widget([""], *estimated filtered cardinality of distinct_col*)
        self.update_widget([""], [self.hll.cardinality()])
