def custom_round(num):
    integer_part = int(num)
    decimal_part = num - integer_part
    if decimal_part >= 0.4:
        return integer_part + 1
    else:
        return integer_part