import matplotlib.pyplot as plt


def visulize_text(sent, data):
    Blues = plt.get_cmap('viridis')

    html = ''
    for item in data:
        token = item['token']
        length = len(token)
        position = sent.index(token)
        token = sent[:position + length]
        token = token.replace('\n', '\\n<br/>')
        sent = sent[position + length:]

        prob = item['prob']
        r, g, b, _ = Blues(prob)
        r, g, b = int(r * 255), int(g * 255), int(b * 255),
        color = '#%02x%02x%02x' % (r, g, b)
        html += '''<span style="color: white; background-color: {}">{}</span>'''.format(color, token)

    with open('output.html', 'w') as fp:
        fp.write(html)


if __name__ == "__main__":
    sent = "Once upon"
    data = [{"token": "Once", "prob": 0}, {"token": " upon", "prob": 0.00031805463368073106}]
    visulize_text(sent, data)
