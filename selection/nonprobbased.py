# -*- coding: utf-8 -*-
# @Author             : GZH
# @Created Time       : 2023-02-03 17:31
# @Email              : guozh29@mail2.sysu.edu.cn
# @Last Modified By   : GZH
# @Last Modified Time : 2023-02-03 17:31

import random

from selection.base import BaseSelect


class RandomSelect(BaseSelect):
    """

    Random selection without probability, inheriting from BaseSelect class.

    """

    def select(self) -> list:
        """

        Client selection.

        Returns:
            List: List of clients selected.

        """

        # Random Sampling.
        return random.sample(
            self.client_list,
            self.select_num
        )


class AllSelect(BaseSelect):
    """

     Full selection without probability, inheriting from BaseSelect class.

    """

    def select(self)-> list:
        """

        Client selection.

        Returns:
            List: List of clients selected.

        """

        # All clients.
        return self.client_list
