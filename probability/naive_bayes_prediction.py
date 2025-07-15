import numpy as np
from ordered_set import OrderedSet

def create_training_data(filename):
    """Initialize the training dataset."""

    with open(filename, "r") as f:
        raw_data = []
        class_names = OrderedSet()

        for line in f:
            data = list(map(str, line.strip().split(',')))
            raw_data.append(data)
            class_names.add(data[-1])
        
        return np.array(raw_data), list(class_names)


def compute_prior_probabilites(train_data, class_names):
    total_samples = len(train_data)
    prior_probs = np.zeros(len(class_names))

    for idx, cls in enumerate(class_names):
        prior_probs[idx] = np.count_nonzero(train_data[:,-1] == cls) / total_samples

    return prior_probs

def compute_conditional_probabilities(train_data, class_names):
    
    n_features = train_data.shape[1] - 1 # Attributes column
    conditional_probs = []
    features_tag = []

    for column in range(n_features):
        unique_values = np.unique(train_data[:, column])
        features_tag.append(unique_values) #Ex: Take every unique data from the column Outlook

        attributes_probability = np.zeros((len(class_names), len(unique_values)))

        for idx1, cls in enumerate(class_names):
            results_in_cls = train_data[train_data[:,-1] == cls]
            count_cls = len(results_in_cls)
            for idx2, feature in enumerate(unique_values):
                results_in_feature_if_cls = results_in_cls[results_in_cls[:, column] == feature]
                count_feature_if_cls = len(results_in_feature_if_cls)

                
                attributes_probability[idx1][idx2] = count_feature_if_cls / count_cls

        conditional_probs.append(attributes_probability)
        
    return conditional_probs, features_tag


def get_feature_index(feature_values, feature_value):
    for idx, feature in enumerate(feature_values):
        if feature_value == feature:
            return idx

def train_naive_bayes(train_data, class_names):
    prior_probabilites = compute_prior_probabilites(train_data, class_names)
    condition_probabilities, features_tag = compute_conditional_probabilities(train_data, class_names)

    return prior_probabilites, condition_probabilities, features_tag

def predict(predict_set, class_names, prior_probabilites, condition_probabilities, features_tag):

    '''
    predict_set =  ['Sunny', 'Cool', 'High', 'Strong']
    -> Find P(Yes | predict_set) -> Find P(predict_set | Yes) * P(Yes) / P(product_set) (<- no needed P(Product_set
    because it is the same for every class, we always divide by it so no need))
    -> P(Yes | predict_set) vs P(NO | predict_set)

    We take P(predict_set | Yes) * P(Yes) vs P(predict_set | No) * P(No)
    
    We assume that every attributes is independent
    -> P(Predict_set | Yes) = Product( P(Predict_set[i] | Yes) for all i in len(Product_set))
    '''   

    class_probs = []
    for cls_idx, cls in enumerate(class_names):
        probability_cls = prior_probabilites[cls_idx]

        for attribute_idx, attribute in enumerate(predict_set):
            feature_idx = get_feature_index(features_tag[attribute_idx], attribute)
            probability_cls *= condition_probabilities[attribute_idx][cls_idx][feature_idx]
    
        class_probs.append(probability_cls)

    total_prob = sum(class_probs)
    
    if total_prob > 0:
        normal_probs = [p / total_prob for p in class_probs]
    else:
        normal_probs = [0.5, 0.5]

    predict_idx = np.argmax(class_probs)
    prediction = class_names[predict_idx]

    prob_dict = {}

    for i in range(len(class_names)):
        prob_dict[class_names[i]] = normal_probs[i].item()

    return prediction, prob_dict, class_probs

#train

data, class_names = create_training_data("./traffic.txt")

prior_probabilites, condition_probabilities, features_tag = train_naive_bayes(data, class_names)

prediction, normalized_probability, class_probs = predict(['Weekday', 'Winter', 'High', 'Heavy'], 
                                                class_names, 
                                                prior_probabilites, 
                                                condition_probabilities, 
                                                features_tag)

for idx, p in enumerate(class_probs):
    print(f"Naive Bayes Probability: P({class_names[idx]}) ~ {p}")

print("------------------")

for attribute, p in normalized_probability.items():
    print(f"Normalized Probability: P({attribute}) = {p}")

print("------------------")

print("Final prediction:", prediction)



