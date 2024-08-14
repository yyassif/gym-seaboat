def _render_indicators(self, W, H):
    s = W / 40.0
    h = H / 40.0
    color = (0, 0, 0)
    polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
    pygame.draw.polygon(self.surf, color=color, points=polygon)

    def vertical_ind(place, val):
        return [
            (place * s, H - (h + h * val)),
            ((place + 1) * s, H - (h + h * val)),
            ((place + 1) * s, H - h),
            ((place + 0) * s, H - h),
        ]

    def horiz_ind(place, val):
        return [
            ((place + 0) * s, H - 4 * h),
            ((place + val) * s, H - 4 * h),
            ((place + val) * s, H - 2 * h),
            ((place + 0) * s, H - 2 * h),
        ]

    assert self.boat is not None
    true_speed = np.sqrt(
        np.square(self.boat.hull.linearVelocity[0])
        + np.square(self.boat.hull.linearVelocity[1])
    )

    # simple wrapper to render if the indicator value is above a threshold
    def render_if_min(value, points, color):
        if abs(value) > 1e-4:
            pygame.draw.polygon(self.surf, points=points, color=color)

    render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))
    # ABS sensors
    render_if_min(
        self.boat.wheels[0].omega,
        vertical_ind(7, 0.01 * self.boat.wheels[0].omega),
        (0, 0, 255),
    )
    render_if_min(
        self.boat.wheels[1].omega,
        vertical_ind(8, 0.01 * self.boat.wheels[1].omega),
        (0, 0, 255),
    )
    render_if_min(
        self.boat.wheels[2].omega,
        vertical_ind(9, 0.01 * self.boat.wheels[2].omega),
        (51, 0, 255),
    )
    render_if_min(
        self.boat.wheels[3].omega,
        vertical_ind(10, 0.01 * self.boat.wheels[3].omega),
        (51, 0, 255),
    )

    render_if_min(
        self.boat.wheels[0].joint.angle,
        horiz_ind(20, -10.0 * self.boat.wheels[0].joint.angle),
        (0, 255, 0),
    )
    render_if_min(
        self.boat.hull.angularVelocity,
        horiz_ind(30, -0.8 * self.boat.hull.angularVelocity),
        (255, 0, 0),
    )




    def _start(self, contact):
        print("Begin Contact")
        fixtureA = contact.fixtureA
        fixtureB = contact.fixtureB
        objectA = fixtureA.body.userData
        objectB = fixtureB.body.userData

        if not objectB:
            return

        if (objectA == 'fastboat' and 'cage' in objectB):
            self.env.cages[int(objectB.split('_')[1])].activate()
            self.env.reward += 1000.0 / len(self.env.cages)
            self.env.cage_visited_count += 1

            # Frame is considered completed if enough % of the cages were captured
            if (self.env.cage_visited_count / len(self.env.cages) > self.fishing_complete_percent):
                self.env.game_over = True

def _contacts(self, contact, begin):
    pass
    # print("Begin Contact")
    # fixtureA = contact.fixtureA
    # fixtureB = contact.fixtureB
    # objectA = fixtureA.body.userData
    # objectB = fixtureB.body.userData
    # cage_userData = None
    # boat_userData = None
    # cage = None
    # boat = None
    # u1 = contact.fixtureA.body
    # u2 = contact.fixtureB.body
    # if objectA and "cage" in objectA:
    #     cage = u1
    #     boat = u2
    #     cage_userData = u1.userData
    #     boat_userData = u2.userData
    # if u2 and "cage" in u2:
    #     cage = u2
    #     boat = u1
    #     cage_userData = u2.userData
    #     boat_userData = u1.userData
    # if not cage_userData:
    #     return

    # print("Contact", cage_userData, boat_userData)
    # print("Contact", cage, boat)

    # inherit cage color from env
    # cage.color = self.env.cage_color / 255
    # if not boat_userData or "cages" not in boat_userData.__dict__:
    #     return
    
    # if begin:
    #     print("Contact begin")
    #     # instance = cage.index
    #     print(cage_userData)
    #     boat.cages.add(cage_userData)
        # Help with a code activating a cage in contact




        
        # if self.env.boat == cage and not cage.is_active:
        #     print("Cage activated")
        #     self.env.cages[int(cage.split('_')[1])].activate()
        #     self.env.reward += 1000.0 / len(self.env.cages)
        #     self.env.cage_visited_count += 1

        #     # Lap is considered completed if enough % of the track was covered
        #     if (self.env.cage_visited_count / len(self.env.cages) > self.fishing_complete_percent):
        #         self.env.terminated = True
    # else:
    #     boat.cages.remove(cage)



    # def _end(self):
    #     print("End Contact")
    #     fixtureA = contact.fixtureA
    #     fixtureB = contact.fixtureB
    #     objectA = fixtureA.body.userData
    #     objectB = fixtureB.body.userData

    #     if not objectB:
    #         return

    #     if (objectA == 'fastboat' and 'cage' in objectB):
    #         self.env.cages[int(objectB.split('_')[1])].activate()
    #         self.env.reward += 1000.0 / len(self.env.cages)
    #         self.env.cage_visited_count += 1

    #         # Frame is considered completed if enough % of the cages were captured
    #         if (self.env.cage_visited_count / len(self.env.cages) > self.fishing_complete_percent):
    #             self.env.game_over = True
