from st_library import HousePricePredictor

hpp = HousePricePredictor()
assert hpp.predict_score(112.3, 3.5, 21.5, 'a') == 4936170.078752257
assert hpp.predict_score(112.3, 3.5, 21.5, 'b') == 5012425.812066328

print("Everything works great (no exceptions raised during assert)")
# Todo: Assert that incorrect type raises correct exception (unit tests)
# Todo: Assert all kind of bad input
# Todo: Replace print with more robust logger (with possibility to write
# to file / ELK / console output etc.)
# Todo: Add integration test
# Todo: Add Git
# Todo: Add CI / CD
# Todo: Dockerize
# Todo: Code Review :)
