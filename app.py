import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

st.title('Market Basket Analysis using Apriori Algorithm')
uploaded_file = st.file_uploader('Upload your CSV file', type=["csv"])
dataset_type = st.radio('Select the type of Dataset', ('Binary matrix', 'Transactional Format'))

# Input for min support and confidence
min_support = st.number_input('Enter minimum support value', min_value=0.0, max_value=1.0, value=0.1)
min_confidence = st.number_input('Enter minimum confidence value', min_value=0.0, max_value=1.0, value=0.1)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Here is a preview of your Dataset')
    st.write(df)

    if dataset_type == 'Binary matrix':
        st.write('You selected a Binary matrix dataset')
        df = df.applymap(lambda x: 1 if x == True or x == 'True' else 0)
        st.write('Transformed Binary Matrix:')
        st.write(df.head())

        # Compute frequent itemsets
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

        # Save frequent itemsets DataFrame to a pickle file
        frequent_itemsets.to_pickle('frequent_itemsets.pkl')

        # Load the frequent itemsets from the pickle file
        frequent_itemsets = pd.read_pickle('frequent_itemsets.pkl')

        # Convert frozenset to string for display
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))

        # Display the frequent itemsets
        st.write("Frequent Itemsets:")
        st.write(frequent_itemsets)

        # Manually create antecedents and consequents
        def create_antecedent_consequent(itemset):
            itemset_list = list(itemset)
            if len(itemset_list) >= 2:
                return [(set(itemset_list[:i] + itemset_list[i+1:]), {itemset_list[i]}) for i in range(len(itemset_list))]
            else:
                return []

        # Create association rules
        rules = []
        for _, row in frequent_itemsets.iterrows():
            itemset = row['itemsets']
            for antecedent, consequent in create_antecedent_consequent(set(itemset.split(', '))):
                rules.append({'antecedents': antecedent, 'consequents': consequent, 'support': row['support']})

        # Convert rules to DataFrame
        association_rules_df = pd.DataFrame(rules)

        if not association_rules_df.empty:
            # Convert sets to strings for display
            association_rules_df['antecedents'] = association_rules_df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
            association_rules_df['consequents'] = association_rules_df['consequents'].apply(lambda x: ', '.join(sorted(x)))

            # Display association rules
            st.write("Association Rules:")
            st.write(association_rules_df)
        else:
            st.write("No association rules were generated with the provided parameters.")
    if dataset_type == 'Transactional Format':
        st.write('You selected a Transactional Format dataset')

         # Ensure all values are strings
        df = df.applymap(str)
        
        # Convert DataFrame to a list of transactions
        transactions = df.apply(lambda x: x.dropna().tolist(), axis=1).tolist()
        

        encoder = TransactionEncoder()
        encoded_array = encoder.fit_transform(transactions)
        df_encoded = pd.DataFrame(encoded_array,columns=encoder.columns_)

        st.write('Transformed Transactional format')
        st.write(df_encoded)

         # Compute frequent itemsets
        frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

        # Save frequent itemsets DataFrame to a pickle file
        frequent_itemsets.to_pickle('frequent_itemsets.pkl')

        # Load the frequent itemsets from the pickle file
        frequent_itemsets = pd.read_pickle('frequent_itemsets.pkl')

        # Convert frozenset to string for display
        frequent_itemsets['itemsets'] = frequent_itemsets['itemsets'].apply(lambda x: ', '.join(sorted(list(x))))

        # Display the frequent itemsets
        st.write("Frequent Itemsets:")
        st.write(frequent_itemsets)


        def  create_antecedent_consequent(itemset):
            itemset_list = list(itemset)
            if len(itemset_list)>=2:
                return[(set(itemset_list[:i] + itemset_list[i+1:]),{itemset_list[i]}) for i in range(len(itemset_list))]
            else:
                return []
            
        rules = []
        for _,row in frequent_itemsets.iterrows():
                itemset = row['itemsets']
                for antecedent, consequent in create_antecedent_consequent(set(itemset.split(', '))):
                 rules.append({'antecedents': antecedent, 'consequents': consequent, 'support': row['support']})

            # Convert rules to DataFrame
        association_rules_df = pd.DataFrame(rules)

             # Convert sets to strings for display
        if not association_rules_df.empty:
          association_rules_df['antecedents'] = association_rules_df['antecedents'].apply(lambda x: ', '.join(sorted(x)))
          association_rules_df['consequents'] = association_rules_df['consequents'].apply(lambda x: ', '.join(sorted(x)))

             # Display association rules
       
          st.write("Association Rules:")
          st.write(association_rules_df)
        else:
              st.write("No association rules were generated with the provided parameters.")
        