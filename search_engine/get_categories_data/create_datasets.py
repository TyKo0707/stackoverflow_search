from data_from_site import CategoryDataset

categories = ['c%23', 'c%2b%2b', 'javascript web', 'javascript', 'python', 'java', 'php', 'android', 'frontend ',
              'jquery', 'iOS', 'database', 'r', 'c', 'asp.net', 'ruby', '.net', 'django', 'angularjs', 'reactjs',
              'regex', 'data-science', 'ruby', 'linux', 'spring', 'windows', 'git', 'macos', 'visual-studio', 'scala',
              'perl', 'rest', 'algorithm', 'excel', 'exception']

i = 0

while i < 1:
    c_type = 'tag' if ' ' not in categories[i] else 'query'
    c = CategoryDataset(categories[i], 1000, c_type)
    try:
        c.create_and_save_dataset()
    except TypeError:
        print('Please enter captcha for ', categories[i])
        input('Input "ok": ')
        continue
    i += 1
    print(c.df.head())
