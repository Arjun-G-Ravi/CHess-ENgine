import tiktoken
import json
import torch

encoder = tiktoken.encoding_for_model('gpt2')

# itoi = None
# itoi2 = None
# idx1toidx2 = None
# idx2toidx1 = None

class PrepData():

    def __init__(self, text):
        
        self.encodings = encoder.encode(text)
        self.vocabs = sorted(list(set(self.encodings)))
        self.itoi = {idx:i for i, idx in enumerate(self.vocabs)}
        self.itoi2 = {i:idx for i, idx in enumerate(self.vocabs)}
        self.vocab_size = len(self.vocabs)


class Tokenizer(PrepData):

    def __init__(self,text) -> None:
        super().__init__(text)
        self.encoder = tiktoken.encoding_for_model('gpt2')
        self.idx1toidx2 = lambda x: [self.itoi[k] for k in x]
        self.idx2toidx1 = lambda x: [self.itoi2[k] for k in x]
        
    def __call__(self, text : str)->list:

        out = self.encoder.encode(text)
        out = self.idx1toidx2(out)
        return out
    def decode(self, tokens : list)-> str:

        out = self.idx2toidx1(tokens)
        out = self.encoder.decode(out)
        return out

class MakeData(Tokenizer):

    def __init__(self, text,context_length : int, batch_size : int) -> None:
        super().__init__(text)
        data = torch.tensor(self.idx1toidx2(encoder.encode(text)), dtype=torch.long)

        n = int(0.9*len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]
        self.context_length = context_length
        self.batch_size = batch_size
    def get_batch(self,split = 'train'):
        data  = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - self.context_length, (self.batch_size,))
        x = torch.stack([data[i:i+self.context_length] for i in ix])
        y = torch.stack([data[i+1:i+self.context_length+1] for i in ix])
        return x, y

if __name__ == '__main__':

    text = """how are you. ! paid for Flickr in 2005. Mark Zuckerberg said Facebook was "committed to building and growing Instagram independently." According to Wired, the deal netted Systrom $400 million.In November 2012, Instagram launched website profiles, allowing anyone to see user feeds from a web browser with limited functionality, as well as a selection of badges, and web widget buttons to link to profiles.Since the app's launch it had used the Foursquare API technology to provide named location tagging. In March 2014, Instagram started to test and switch the technology to use Facebook Places.


=== 2015–2017: Redesign and Windows app ===
In June 2015, the desktop website user interface was redesigned to become more flat and minimalistic, but with more screen space for each photo and to resemble the layout of Instagram's mobile website. Furthermore, one row of pictures only has three instead of five photos to match the mobile layout. The slideshow banner on the top of profile pages, which simultaneously slide-showed seven picture tiles of pictures posted by the user, alternating at different times in a random order, has been removed. In addition, the formerly angular profile pictures became circular.
In April 2016, Instagram released a Windows 10 Mobile app, after years of demand from Microsoft and the public to release an app for the platform. The platform previously had a beta version of Instagram, first released on November 21, 2013, for Windows Phone 8. The new app added support for videos (viewing and creating posts or stories, and viewing live streams), album posts and direct messages. Similarly, an app for Windows 10 personal computers and tablets was released in October 2016. In May, Instagram updated its mobile website to allow users to upload photos, and to add a "lightweight" version of the Explore tab.On May 11, 2016, Instagram revamped its design, adding a black-and-white flat design theme for the app's user interface, and a less skeuomorphistic, more abstract, "modern" and colorful icon. Rumors of a redesign first started circulating in April, when The Verge received a screenshot from a tipster, but at the time, an Instagram spokesperson simply told the publication that it was only a concept.On December 6, 2016, Instagram introduced comment liking. However, unlike post likes, the user who posted a comment does not receive notifications about comment likes in their notification inbox. Uploaders can optionally decide to deactivate comments on a post.The mobile web front end allows uploading pictures since May 4, 2017. Image filters and the ability to upload videos were not introduced then.On April 30, 2019, the Windows 10 Mobile app was discontinued, though the mobile website remains available as a progressive web application (PWA) with limited functionality. The app remains available on Windows 10 computers and tablets, also updated to a PWA in 2020.


=== 2018–2019: IGTV, removal of the like counter, management changes ===
To comply with the GDPR regulations regarding data portability, Instagram introduced the ability for users to download an archive of their user data in April 2018.IGTV launched on June 20, 2018, as a standalone video application.
On September 24, 2018, Krieger and Systrom announced in a statement they would be stepping down from Instagram. On October 1, 2018, it was announced that Adam Mosseri would be the new head of Instagram.During Facebook F8, it was announced that Instagram would, beginning in Canada, pilot the removal of publicly displayed "like" counts for content posted by other users. Like counts would only be visible to the user who originally posted the content. Mosseri stated that this was intended to have users "worry a little bit less about how many likes they're getting on Instagram and spend a bit more time connecting with the people that they care about." It has been argued that low numbers of likes in relativity to others could contribute to a lower self-esteem in users. The pilot began in May 2019, and was extended to 6 other markets in July. The pilot was expanded worldwide in November 2019. Also in July 2019, Instagram announced that it would implement new features designed to reduce harassment and negative comments on the service.In August 2019, Instagram also began to pilot the removal of the "Following" tab from the app, which had allowed users to view a feed of the likes and comments made by users they follow. The change was made official in October, with head of product Vishal Shah stating that the feature was underused and that some users were "surprised" when they realized their activity was being surfaced in this manner.In October 2019, Instagram introduced a limit on the number of posts visible in page scrolling mode unless logged in. Until this point, public profiles had been available to all users, even when not logged in. Following the change, after viewing a number of posts a pop-up requires the user to log in to continue viewing content.That"""

    # PrepData(text)
    tokenizer = Tokenizer(text)
    # print(itoi)
    out = tokenizer("how are you")
    print(out)

    out = tokenizer.decode(out)
    print(out)

    mkdata = MakeData(text, 100, 3)
    x,y = mkdata.get_batch()
    print(x.shape, y.shape)