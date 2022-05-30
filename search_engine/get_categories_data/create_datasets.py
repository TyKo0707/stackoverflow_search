from data_from_site import CategoryDataset

categories = ['c%23', 'c%2b%2b', 'javascript', 'python', 'java', 'php', 'android', 'api'
              'jquery', 'iOS', 'database', 'r', 'c', 'asp.net', 'ruby', '.net', 'django', 'angularjs', 'reactjs',
              'regex', 'data-science', 'linux', 'spring', 'windows', 'git', 'macos', 'visual-studio', 'scala',
              'perl', 'rest', 'algorithm', 'excel', 'html', 'css']

c = CategoryDataset(categories)
c.create_and_save_dataset()
