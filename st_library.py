import json
import math
import numpy as np


class HousePricePredictor(object):
    # Please note this is compiling on Mac, you may have to change the
    # directory format for running on Windows.
    def __init__(self, directory='./'):
        self.models = {'a': self.parse_model(directory + 'jsmn-type-a.json'),
                       'b': self.parse_model(directory + 'jsmn-type-b.json')}

    def parse_model(self, model_filename):
        model = json.load(open(model_filename))
        print('loading file: {}'.format(model_filename))
        parameters_array = []
        model['parameters'] = {parameter['feature_name']: parameter for
                               parameter
                               in model['parameters']}
        for k, v in model['parameters'].items():
            # Add Standard Deviation for Z-Score computation.
            v['std_dev'] = math.sqrt(v['variance'])
            v.pop('feature_name', None)
            parameters_array.append(v['parameter'])

        # Ordered in the same order as X
        model['beta'] = np.array(parameters_array)
        print("Model parameters: {}\n".format(model))
        return model

    def compute_z_score(self, value, parameter):
        """
        Args:
                value (double): Original value to normalize
                parameter (dict): parameter from the model, containing std dev
                and mean

        Returns:
                value_z_score (double): Z-score of the value, as per the model
                parameters.
        """
        return (value - parameter['mean']) / parameter['std_dev']

    def predict_score(self, area, rooms, years_since_build, type):
        """
        Args:
                area (double): A feature, area of the house measures in m^2 (112.3)
                rooms (double): A feature, number of rooms (3.5)
                years_since_build (double): A feature, number of years since
                building the house (21.5)
                type (String): The required model type

        Returns:
                house_price (double): The prediction --> Regression problem !
        """
        if type not in ['a', 'b']:
            raise ValueError('Model type is unsupported')

        if area < 0 or rooms < 0 or years_since_build < 0:
            raise ValueError('Illegal parameters (should be positive)')

        print('Predicting score with model: "{}"'.format(type))
        model = self.models[type]

        model_parameters = model['parameters']
        area_z_score = self.compute_z_score(area, model_parameters['area'])
        rooms_z_score = self.compute_z_score(rooms, model_parameters['rooms'])
        years_since_build_z_score = \
            self.compute_z_score(years_since_build,
                                 model_parameters['years_since_build'])

        X = np.array([area_z_score, rooms_z_score, years_since_build_z_score])
        print("X matrix: {}".format(X))
        print("Model Beta: {}".format(model['beta']))
        print("Model offset: {}".format(model['offset']))
        house_price = X.T.dot(model['beta']) + model['offset']
        print("House price (X.T.dot(Beta) + offset): {}\n".format(house_price))

        return house_price
